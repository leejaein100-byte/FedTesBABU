# -*- coding: utf-8 -*-
import os
import shutil
import json
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import push
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
from train_and_test_Gr import *
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Fedtes_data_dist import *
from utils.sampling import *
from utils.utils import *
#from Stiefel_model import *
from Gr_model import *


def create_push_dataloader(args, dataset, server_idx, dict_users, device):
    """
    Create a dataloader for prototype pushing using training data from all clients.
    """
    all_X_train = []
    all_y_train = []
    
    # Collect training data from all clients
    for client_idx in range(args.num_users):
        X_train, y_train = load_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
        all_X_train.append(X_train)
        all_y_train.append(y_train)
    
    # Combine all training data
    combined_X = torch.cat(all_X_train, dim=0)
    combined_y = torch.cat(all_y_train, dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize: x_denorm = (x_norm * std) + mean
    combined_X_denorm = combined_X * std + mean
    
    # Clamp to [0, 1] range to ensure valid image data
    #combined_X_denorm = torch.clamp(combined_X_denorm, 0.0, 1.0)    
    # Create dataset
    push_dataset = torch.utils.data.TensorDataset(combined_X_denorm, combined_y)
    
    # Create dataloader (no normalization here as push expects unnormalized data in [0,1])
    push_dataloader = torch.utils.data.DataLoader(
        push_dataset,
        batch_size=args.local_bs,  # Use same batch size as training
        shuffle=False,  # Don't shuffle for consistent prototype selection
        num_workers=0,  # Set to 0 to avoid potential issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return push_dataloader

def conduct_prototype_push(args, model, dataset, server_idx, dict_users, device, 
                          epoch, model_dir):

    push_dataloader = create_push_dataloader(args, dataset, server_idx, dict_users, device)
    
    # Set up directories for saving prototypes
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    
    # Ensure model is on correct device
    model = model.to(device)

    if hasattr(model, 'module'):  # DataParallel wrapped
        model_parallel = model.module
    else:  # Regular model
        model_parallel = model
    
    # Conduct prototype push
    push.push_prototypes_top3(
        dataloader=push_dataloader,
        prototype_network_parallel=model_parallel,
        class_specific=True,  # Assuming class-specific prototypes
        #preprocess_input_function=preprocess_input_function,  # Use your preprocessing
        preprocess_input_function=None,
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=img_dir,
        epoch_number=epoch,
        prototype_img_filename_prefix=prototype_img_filename_prefix,
        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
        save_prototype_class_identity=True,
        log=print, prototype_activation_function_in_numpy=None)

def save_settings(args, settings_dir):
    """
    Save command line arguments to a settings file in the same directory as the model.
    
    Parameters:
    args: argparse.Namespace
        The parsed command line arguments
    """
    settings_dict = {
    'iid': args.iid,
    'num_channels': args.num_channels,
    'server_id_size': args.server_id_size,
    'local_bs': args.local_bs,
    'num_users': args.num_users,
    'arch': args.arch,
    'dataset': args.dataset,
    'SL_epochs': args.SL_epochs,
    'fine_tune_epochs': args.fine_tune_epochs,
    'alpha': args.alpha,
    'use_bbox': args.use_bbox,
    'temp': args.temp,
    'warmup_ep':args.warmup_ep,
    'hyperparam': args.hyperparam}

    # Save settings to JSON file
    settings_file = os.path.join(settings_dir, 'settings.json')
    with open(settings_file, 'w') as f:
        json.dump(settings_dict, f, indent=4)


def get_available_device():
    
    if not torch.cuda.is_available():
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()
    
    for gpu_id in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - allocated_memory
        
        # Check if more than half of the memory is free
        if free_memory > total_memory / 2:
            return torch.device('cuda:{}'.format(gpu_id))

    # If no GPU with >50% free memory is found, return the GPU with the most free memory
    max_free_memory = 0
    best_gpu_id = 0

    for gpu_id in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - allocated_memory
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu_id = gpu_id

    return torch.device(f'cuda:{best_gpu_id}')


def load_model_and_push(model_path, args):
    """
    Load a trained model from the specified path and conduct prototype pushing.
    """
    device = get_available_device()
    model_dir = os.path.dirname(model_path)
    #log, logclose = create_logger(log_filename=os.path.join(model_dir, 'push.log'))    
    dict_users, server_idx, dataset = setup_datasets(args)
    dataset_name = args.dataset
    base_architecture = args.arch
    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    prototype_per_class = settings_CUB.prototype_per_class
    
    # Construct model
    model = construct_TesNet(
        base_architecture=base_architecture,
        prototype_per_class=prototype_per_class,
        dataset=dataset_name,
        pretrained=True,
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=prototype_activation_function,
        add_on_layers_type=add_on_layers_type
    )
    
    # Load trained weights
    #log(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Conduct prototype push
    conduct_prototype_push(
        args=args,
        model=model,
        dataset=dataset,
        server_idx=server_idx,
        dict_users=dict_users,
        device=device,
        epoch='final',  # or specify epoch number
        model_dir=model_dir,
        #log=log
    )
    
    #logclose()
def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '/'+ str(args.iid) + '/'+ str(start_time) + '/' + 'FedTesBABU NonCayleyT'

    if os.path.exists(model_dir) is True:
        shutil.rmtree(model_dir)
    makedir(model_dir)
    log_directory = f"checkpoints/{str(args.iid)}/'FedTesBABU NonCayleyT_CUB'/{start_time+'tuning with interval without augmented dataset'}/"
    writer = SummaryWriter(log_dir=log_directory)    
    shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    coefs = settings_CUB.coefs
    num_train_epochs = args.num_train_epochs
    #user_datasets= load_cropped_CUB_random_distribution(num_users=args.num_users, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True)
    prototype_per_class = settings_CUB.prototype_per_class
    dict_users_train, server_id, dataset=setup_datasets(args)
    net_glob = construct_TesNet(base_architecture=base_architecture, prototype_per_class=prototype_per_class, dataset=dataset_name,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    save_settings(args, model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    current_script_path = os.path.abspath(__file__)
    
    filename = os.path.basename(current_script_path)
    log(f"Currently running: {filename}")
    update_keys = [k for k in net_glob.state_dict().keys() 
                      if 'linear' not in k and 'last_layer' not in k]
    device = get_available_device()
    
    # Initialize clients list first
    clients = []
    for _ in range(args.num_users):
        clients.append(copy.deepcopy(net_glob))
    
    # Prepare combined test data
    X_test = []
    y_test = []
    for client_idx in range(args.num_users):
        #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
        X_temp_test, y_temp_test = load_data(args, dataset, server_id, client_idx, dict_users_train, train = False, private = True)
        X_test.append(X_temp_test)
        y_test.append(y_temp_test)
    
    X_test = torch.cat(X_test)
    y_test = torch.cat(y_test)
    net_glob = net_glob.to(device)
    net_glob.prototype_vectors = net_glob.prototype_vectors.to(device)
    
    # Main training loop
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))
        return_dict = {}
        clients_state_list = []
        
        # Train each client
        for client_idx in range(args.num_users):
            clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
            #X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
            X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
            #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
            clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train, 
                                                 is_train=True, body_train=True, coefs=coefs, log=log)
            return_dict[client_idx] = loss_dict
            clients_state_list.append(clients[client_idx].module.state_dict())
            clients[client_idx] = clients[client_idx].module
            X_client_test,y_client_test =  load_data(args, dataset, server_id, client_idx, dict_users_train, train = False, private = True)
            #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
            client_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
            log(f'Client {client_idx} test accuracy before aggregation: {client_test_results["accu"]}')
            log(f'Client {client_idx} loss values before aggregation: {loss_dict}')

        # Federated averaging of non-prototype parameters
        net_glob.load_state_dict(FedAvg(clients_state_list, net_glob.state_dict()))
        
        # Consensus update for prototype vectors
        temp2 = []
        Gm_manifold = []
        
        # Collect prototype vectors from all clients and global model
        for client_idx in range(args.num_users):
            Gm_manifold.append(torch.unsqueeze(clients[client_idx].prototype_vectors.data.to(device), 0).to(device))        
        Gm_manifold = torch.cat(Gm_manifold, dim=0)

        # For a 4D tensor, permute dimensions
        if Gm_manifold.dim() == 4:
            Gm_manifold = Gm_manifold.permute(1, 0, 2, 3)
        else:
            print(f"Warning: Gm_manifold has unexpected shape: {Gm_manifold.shape}")

        for cls in range(num_classes):
            temp1 = consensus_update(Gm_manifold[cls])
            temp2.append(torch.unsqueeze(temp1, 0))
        
        print('frechet mean finished')
        net_glob.prototype_vectors.data = torch.cat(temp2, dim=0)     
        del temp1, temp2

        # Update clients with global model parameters
        for user_idx in range(args.num_users):
            for k in update_keys:
                clients_state_list[user_idx][k] = copy.deepcopy(net_glob.state_dict()[k])    
            clients[user_idx].load_state_dict(clients_state_list[user_idx])
        
        global_test_results = local_test_global_model(args, net_glob, X_test, y_test, coefs)  
        writer.add_scalar('Global/Loss', global_test_results['average loss'], global_step=epoch)
        writer.add_scalar('Global/Accuracy', global_test_results['accu'], global_step=epoch) 
        log('Global model test')
        log(global_test_results)
        
        # Test individual clients on their test data
        for client_idx in range(args.num_users):
            #X_train, y_train, X_client_test, y_client_test = load_data_random(user_datasets, client_idx)
            #X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
            X_client_test, y_client_test = load_data(args, dataset, server_id, client_idx, dict_users_train, train = False, private = True)
            #X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
            client_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
            writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
            writer.add_scalar(f'Client_{client_idx}/Loss', client_test_results['average loss'], global_step=epoch)
            log(f'Client {client_idx} test accuracy: {client_test_results["accu"]}')
            
    log('Starting fine-tuning phase')
    client_ft_results = {}
    for client_idx in range(args.num_users):
        client_ft_results[client_idx] = []
        X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
        X_client_test, y_client_test = load_data(args, dataset, server_id, client_idx, dict_users_train, train = False, private = True)
        #X_client_test, y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = False)
        for fine_tune_epoch in range(args.fine_tune_epochs):
            clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
            clients[client_idx], loss_dict = fine_tune_train(args, client_idx, clients[client_idx], X_train, y_train, 
                is_train=True, body_train=False, coefs=coefs, log=print)
            clients[client_idx] = clients[client_idx].module
            ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
            client_ft_results[client_idx].append(ft_test_results['accu'])
            
            # Log to TensorBoard
            writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
            writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
            log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')
            log(loss_dict)

    final_model_path = os.path.join(model_dir, 'final_model.pth')
    torch.save(net_glob.state_dict(), final_model_path)
    log(f'Final model saved at {final_model_path}')
    
    final_client_model_path = os.path.join(model_dir, 'client_model.pth')
    torch.save(clients[0].state_dict(), final_client_model_path)
    log(f'Final model saved at {final_client_model_path}')
    
    # Final prototype push
    conduct_prototype_push(
        args=args,
        model=net_glob,
        dataset=dataset,
        server_idx=server_id,
        dict_users=dict_users_train,
        device=device,
        epoch='final',
        model_dir=model_dir
    )
    conduct_prototype_push(
        args=args,
        model=clients[0],
        dataset=dataset,
        server_idx=server_id,
        dict_users=dict_users_train,
        device=device,
        epoch='final',
        model_dir=os.path.join(model_dir,'clients'))
    
    logclose()
if __name__ == "__main__":
    main()