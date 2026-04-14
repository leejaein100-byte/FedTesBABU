import os
import shutil
from torch.utils.data import DataLoader, TensorDataset
import json
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import time
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import _load_StanfordDogs
from utils.Fedtes_data_dist import load_cropped_CUB
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
from Centralized_push import *
from centralized_model import *
import jilee.FedTesBABU.train_and_test_lr_schedule as tnt
from util import save
from util.log import create_logger
from utils.utils import *
import settings_CUB_Centralized
from util.preprocess import mean, std, preprocess_input_function
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a tuple of the sample and its corresponding label
        return self.X[index], self.y[index]

def save_settings(args, settings_dir):

    settings_dict = {
    #'iid': args.iid,
    'num_channels': args.num_channels,
    'arch': args.arch,
    'dataset': args.dataset,
    #'SL_epochs': args.SL_epochs,
    'fine_tune_epochs': args.fine_tune_epochs,
    'push_interval': args.push_interval,
    'fine_tune_interval': args.fine_tune_interval,
    'prototype_per_class': args.prototype_per_class}

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

def main():

    args = settings_CUB_Centralized.args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.Device_id
    num_classes = args.num_classes
    img_size = settings_CUB_Centralized.img_size
    add_on_layers_type = settings_CUB_Centralized.add_on_layers_type
    #prototype_shape = settings_CUB_Centralized.prototype_shape
    prototype_shape = (num_classes * args.prototype_per_class, args.num_channels, 1, 1)
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
    num_train_epochs = args.num_train_epochs
    num_warm_epochs = settings_CUB_Centralized.num_warm_epochs
    push_start = settings_CUB_Centralized.push_start
    #push_epochs = settings_CUB_Centralized.push_epochs
    push_epochs = [i for i in range(num_train_epochs) if i % args.fine_tune_interval == 0]
    #else:
    #   raise Exception("there are no settings file of datasets {}".format(dataset_name))

    base_architecture = args.arch

    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    #best acc
    all_global_accs = [] # [seed1_acc, seed2_acc, seed3_acc]
    device = torch.device(f'cuda:0')
    dataset_name = args.dataset
    # train the model
    for trial in range(3):
        initial_seed = args.seed
        current_seed = initial_seed + trial
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        model_dir = './saved_models/' + 'centralized/' + dataset_name+'/' + base_architecture + '/' + str(start_time) + '/'    #+ args.times + '/'
        if os.path.exists(model_dir) is True:
            shutil.rmtree(model_dir)
        makedir(model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB_Centralized.py'), dst=model_dir)
        img_dir = os.path.join(model_dir, 'img')
        log_directory = f"checkpoints/{start_time+ dataset_name + ' Centralized Model without augmented dataset'}/"
        writer = SummaryWriter(log_dir=log_directory)    
        makedir(img_dir)

        normalize = transforms.Normalize(mean=mean,std=std)
        log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

        if dataset_name == 'Stanford_dog':
            all_images, labels =  _load_StanfordDogs()
            total_samples = len(all_images)
            print(f"Total samples loaded: {total_samples}")


            train_indices, test_indices = train_test_split(list(range(total_samples)),train_size=args.tr_frac, random_state=current_seed) 
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
        
        elif dataset_name == 'CUB':
            all_images, all_labels = load_cropped_CUB()
            print(f"Dataset loaded. Total images: {len(all_images)}")

            # 2. Create a list of indices to split
            # We will split the indices instead of the data itself for efficiency
            indices = list(range(len(all_labels)))

            # 3. Use train_test_split to get stratified indices
            # We must use .numpy() on the labels for sklearn's stratify function
            print("Splitting data (stratified)...")
            train_indices, test_indices = train_test_split(
                indices,
                test_size=0.2,  # 20% for test set
                stratify=all_labels.numpy(), # This ensures the same label distribution
                random_state=current_seed # for reproducibility
            )

            # 4. Create TensorDatasets for train and test using the split indices
            train_dataset = TensorDataset(
                all_images[train_indices], 
                all_labels[train_indices]
            )
            test_dataset = TensorDataset(
                all_images[test_indices], 
                all_labels[test_indices]
            )

        train_push_dataset = DenormalizedDataset(train_dataset, mean=mean, std=std)
        train_push_loader = torch.utils.data.DataLoader(
                train_push_dataset,
                batch_size=train_push_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False)
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

        class_specific = True
        save_settings(args, model_dir)
        ppnet = construct_TesNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
            

        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)
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
            if epoch == num_train_epochs - 1:
                all_global_accs.append(test_results['accu'])            
        #writer.add_scalar('Global/Loss', test_results['average loss'], global_step=epoch)
        #writer.add_scalar('Global/Accuracy', test_results['accu'], global_step=epoch) 
        #log('Global model test')
        #log(test_results)
        logclose()

    gm_mean = np.mean(all_global_accs) * 100
    gm_std = np.std(all_global_accs) * 100
    #trial_means = [np.mean(seed_accs) for seed_accs in all_local_accs]
    #pm_mean = np.mean(trial_means) * 100
    #pm_std = np.std(trial_means) * 100

# Prepare the data dictionary
    results_to_save = {
        "global_model": {
            "mean": float(gm_mean),
            "std": float(gm_std),
            "raw_accs": [float(x) for x in all_global_accs]
        },
        "settings": {
            "base_seed": int(initial_seed)
        }
    }

    # Define the file path (using the existing model_dir from your code)
    json_file_path = os.path.join(model_dir, 'final_results.json') 

    # Write to JSON file
    with open(json_file_path, 'w') as f:
        json.dump(results_to_save, f, indent=4) 

    print(f"\nResults successfully saved to: {json_file_path}")
    print("\n" + "="*30)
    print(f"FINAL RESULTS ACROSS 3 SEEDS")

if __name__ == "__main__":
    main()