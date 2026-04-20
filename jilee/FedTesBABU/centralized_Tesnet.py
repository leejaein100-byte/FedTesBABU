import os
import shutil
import torch.utils.data
import json
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import re
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import _load_StanfordDogs
from tensorboardX import SummaryWriter
from util.helpers import makedir
from Centralized_push import *
from centralized_model import *
import train_and_test as tnt
from util import save
from util.log import create_logger
from utils.utils import *
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB_Centralized
from sklearn.model_selection import train_test_split

def save_settings(args, settings_dir):

    settings_dict = {
    #'iid': args.iid,
    'num_channels': args.num_channels,
    #'server_id_size': args.server_id_size,
    #'local_bs': args.local_bs,
    #'num_users': args.num_users,
    'arch': args.arch,
    'dataset': args.dataset,
    #'SL_epochs': args.SL_epochs,
    'fine_tune_epochs': args.fine_tune_epochs,
    'push_interval': args.push_interval,
    'fine_tune_interval': args.fine_tune_interval}

    # Save settings to JSON file
    settings_file = os.path.join(settings_dir, 'settings.json')
    with open(settings_file, 'w') as f:
        json.dump(settings_dict, f, indent=4)


class DenormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, normalized_dataset, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.normalized_dataset = normalized_dataset
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __len__(self):
        return len(self.normalized_dataset)
    
    def __getitem__(self, idx):
        image, label = self.normalized_dataset[idx]
        denormalized_image = image * self.std + self.mean
        denormalized_image = torch.clamp(denormalized_image, 0.0, 1.0)
        return denormalized_image, label

# Create push dataloader
    

def main():
    args = settings_CUB_Centralized.args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.Device_id
    num_classes = settings_CUB_Centralized.num_classes
    img_size = settings_CUB_Centralized.img_size
    add_on_layers_type = settings_CUB_Centralized.add_on_layers_type
    prototype_shape = settings_CUB_Centralized.prototype_shape
    prototype_activation_function = settings_CUB_Centralized.prototype_activation_function
    #datasets
    train_batch_size = settings_CUB_Centralized.train_batch_size
    test_batch_size = settings_CUB_Centralized.test_batch_size
    train_push_batch_size = settings_CUB_Centralized.train_push_batch_size
    #optimzer
    joint_optimizer_lrs = settings_CUB_Centralized.joint_optimizer_lrs
    joint_lr_step_size = settings_CUB_Centralized.joint_lr_step_size
    warm_optimizer_lrs = settings_CUB_Centralized.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_CUB_Centralized.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_CUB_Centralized.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_CUB_Centralized.num_train_epochs
    num_warm_epochs = settings_CUB_Centralized.num_warm_epochs
    push_start = settings_CUB_Centralized.push_start
    push_epochs = settings_CUB_Centralized.push_epochs

    #else:
    #   raise Exception("there are no settings file of datasets {}".format(dataset_name))

    base_architecture = args.arch
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
    #model save dir
    dataset_name = args.dataset
    model_dir = './saved_models/' + 'centralized/' + dataset_name+'/' + base_architecture + '/' + str(start_time) + '/'    #+ args.times + '/'

    if os.path.exists(model_dir) is True:
        shutil.rmtree(model_dir)
    makedir(model_dir)

    shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB_Centralized.py'), dst=model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    log_directory = f"checkpoints/{start_time+ dataset_name + ' Centralized Model without augmented dataset'}/"
    writer = SummaryWriter(log_dir=log_directory)    
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    normalize = transforms.Normalize(mean=mean,std=std)

    all_images, labels =  _load_StanfordDogs()
    total_samples = len(all_images)
    print(f"Total samples loaded: {total_samples}")

    # Define train fraction
    train_fraction = 0.8  # Use 80% for training
    train_indices, test_indices = train_test_split(list(range(total_samples)),train_size=args.tr_frac, random_state=args.seed) 
    #train_dataset =  torch.utils.data.DataLoader(
    #   MyDataset(all_images[train_indices]), labels(train_indices), batch_size=train_batch_size, shuffle=True,
    #  num_workers=4, pin_memory=False)   
    #test_dataset =  torch.utils.data.DataLoader(
    #   MyDataset(all_images[test_indcies]), labels(test_indices), batch_size=train_batch_size, shuffle=True,
    #  num_workers=4, pin_memory=False)   

    train_images = all_images[train_indices]
    train_labels = labels[train_indices]
    test_images = all_images[test_indices]
    test_labels = labels[test_indices]

    # Create dataset objects (NOT DataLoaders yet)
    train_dataset = MyDataset(train_images, train_labels)
    test_dataset = MyDataset(test_images, test_labels)
    save_settings(args, model_dir)
    train_push_dataset = DenormalizedDataset(train_dataset, mean=mean, std=std)
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset,
        batch_size=train_push_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  # Pass the dataset object, not the constructor
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,  # Pass the dataset object, not the constructor
        batch_size=test_batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=False
    )
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    #log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    log("backbone architecture:{}".format(base_architecture))
    log("basis concept size:{}".format(prototype_shape))
    # construct the model
    ppnet = construct_TesNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    from settings_CUB_Centralized import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings_CUB_Centralized import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings_CUB_Centralized import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    #best acc
    best_acc = 0
    best_epoch = 0
    best_time = 0

    # train the model
    log('start training')
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))
        #stage 1: Embedding space learning
        #train
        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)

        #test
        test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        writer.add_scalar('Centralized/Loss', train_results['average loss'], global_step=epoch)
        writer.add_scalar('Centralized/Accuracy', test_results['accu'], global_step=epoch)
        log(test_results)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=test_results['accu'],
                                    target_accu=0.70, log=log)
        #stage2: Embedding space transparency, 여기가 바뀌어야할 부분 3/22
        if epoch >= push_start and epoch in push_epochs:
            if epoch % (args.fine_tune_interval * args.push_interval) == 0:
                push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=test_results['accu'],
                                        target_accu=0.70, log=log)
            
        #stage3: concept based classification
            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(args.fine_tune_epochs):
                    log('iteration: \t{0}'.format(i))
                    train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)

                    test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=test_results['accu'],
                                                target_accu=0.70, log=log)
                    writer.add_scalar(f'FineTune/Centralized_Model_Test Accuracy', test_results['accu'], global_step = i + 10*(epoch//10))
                    writer.add_scalar(f'FineTune/Centralized_Model_loss', train_results['average loss'], global_step= i + 10*(epoch//10))
                    log(f'Centralized_Model Fine-tune epoch {i}, accuracy: {test_results["accu"]}')
                    log(test_results)
        
        #writer.add_scalar('Global/Loss', test_results['average loss'], global_step=epoch)
        #writer.add_scalar('Global/Accuracy', test_results['accu'], global_step=epoch) 
        #log('Global model test')
        #log(test_results)
    logclose()

if __name__ == "__main__":
    main()