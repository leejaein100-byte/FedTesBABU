import os
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import push_st
from torch.utils.tensorboard import SummaryWriter
#from St_model import construct_TesNet
from util.helpers import makedir
from train_and_test_St import *
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from utils.sampling import *
from utils.utils import *
#from Stiefel_model import *
#from utils.misc import *

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return a tuple of the sample and its corresponding label
        return self.X[index], self.y[index]

def create_push_dataloader(args, dataset, server_idx, dict_users, device):
    """
    Create a dataloader for prototype pushing using training data from all clients.
    """
    all_X_train = []
    all_y_train = []
    
    # Collect training data from all clients
    for client_idx in range(args.num_users):
        X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train=True)
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

    #if hasattr(model, 'module'):  # DataParallel wrapped
    #    model_parallel = model.module
    #else:  # Regular model
    #    model_parallel = model
    
    # Conduct prototype push
    push_st.push_prototypes(
        dataloader=push_dataloader,
        #prototype_network_parallel=model_parallel,
        prototype_network_parallel=model,
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
    'num_train_epochs': args.num_train_epochs,
    'seed': args.seed,
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
    'use_bbox': args.use_bbox}

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


def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    initial_seed = args.seed
    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    #prototype_shape = settings_CUB.prototype_shape
    prototype_per_class = settings_CUB.prototype_per_class
    prototype_shape = (num_classes*prototype_per_class, args.num_channels,1,1)
    prototype_activation_function = settings_CUB.prototype_activation_function
    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-3, 
        'sub_sep': -1e-7,
    }

    num_train_epochs = args.num_train_epochs
    #user_datasets= load_cropped_CUB_random_distribution(num_users=args.num_users, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True)
    all_global_accs = [] # [seed1_acc, seed2_acc, seed3_acc]
    all_local_accs = []  # [ [c1_acc, c2_acc...], [c1_acc...], [c1_acc...] ]
    all_local_accs_bef_FT = []
    all_univ_local_accs = []
    device = torch.device(f'cuda:0')
    for trial in range(3):
        current_seed = initial_seed + trial
        args.seed = current_seed
        print(f"\n === Starting Trial {trial+1}/3 with Seed: {current_seed} === \n")
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        model_dir = './saved_models/' + 'Stiefel' + '/' + dataset_name+ '/' + base_architecture + '/'+ str(args.iid) + '/'+ str(start_time) + '/'+'FedTesBABU'
        if os.path.exists(model_dir) is True:
            shutil.rmtree(model_dir)
        makedir(model_dir)
        log_directory = f"checkpoints/{str(args.iid)}/'FedTes_Stiefel'/{start_time+'without augmented dataset'}/"
        writer = SummaryWriter(log_dir=log_directory)    
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
        dict_users, server_idx, dataset=setup_datasets(args)
        net_glob = construct_TesNet(base_architecture, dataset=dataset_name,
                                pretrained=True, img_size=img_size,
                                prototype_shape = prototype_shape, 
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
        update_keys = [k for k in net_glob.state_dict().keys() 
                      if 'linear' not in k and 'last_layer' not in k]
        save_settings(args, model_dir)
        log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
        img_dir = os.path.join(model_dir, 'img')
        makedir(img_dir)
        current_script_path = os.path.abspath(__file__)  
        filename = os.path.basename(current_script_path)
        log(f"Currently running: {filename}")
        if hasattr(args, 'load_model_path') and args.load_model_path:
            if os.path.exists(args.load_model_path):
                log(f"Loading pretrained model from: {args.load_model_path}")
                state_dict = torch.load(args.load_model_path, map_location='cpu')
                
                # Load the state dict into the model
                net_glob.load_state_dict(state_dict)
                log("Model state loaded successfully.")
            else:
                log(f"Warning: Model path specified but not found: {args.load_model_path}. Initializing new model.")
        else:
            log("No pretrained model path specified, initializing new model.")
     
    # Initialize clients list first
        clients = []
        for _ in range(args.num_users):
            clients.append(copy.deepcopy(net_glob))
    
        # Prepare combined test data
        X_test = []
        y_test = []
        for client_idx in range(args.num_users):
            #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
            X_temp_test,y_temp_test= load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
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
            
            for client_idx in range(args.num_users):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                #X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
                X_train,y_train=load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= True)
                #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
                clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train, 
                                                    is_train=True, body_train=True, coefs=coefs, log=log)
                return_dict[client_idx] = loss_dict
                clients_state_list.append(clients[client_idx].module.state_dict())
                clients[client_idx] = clients[client_idx].module
                X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
                log(f'Client {client_idx} test accuracy before aggregation: {client_test_results["accu"]}')
                log(f'Client {client_idx} loss values before aggregation: {loss_dict}')

            net_glob.load_state_dict(FedAvg(clients_state_list, net_glob.state_dict()))
            
            all_prototypes = [
                torch.squeeze(clients[idx].prototype_vectors.data.to(device)) 
                for idx in range(args.num_users)]

            # Shape: [num_users, num_classes, dim, C]
            St_manifold = torch.stack(all_prototypes, dim=0)
            # Stack the resulting mean prototypes back together
            # Shape: [num_classes, dim, C]
            #print('Size of St_manifold', St_manifold.size())
            St_manifold =  St_manifold.reshape(args.num_users, net_glob.num_classes, net_glob.prototype_shape[1], net_glob.num_prototypes_per_class)
            St_manifold = St_frechet_mean(St_manifold).reshape(net_glob.num_classes*net_glob.num_prototypes_per_class, net_glob.prototype_shape[1])
            net_glob.prototype_vectors.data= St_manifold.unsqueeze(-1).unsqueeze(-1)
            print('frechet mean finished')

            # 3. Update clients with global model parameters (this part remains the same)
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
                X_client_test, y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
                writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
                writer.add_scalar(f'Client_{client_idx}/Loss', client_test_results['average loss'], global_step=epoch)
                log(f'Client {client_idx} test accuracy: {client_test_results["accu"]}')
                if epoch == num_train_epochs -1:
                    all_local_accs_bef_FT.append(client_test_results['accu'])
                    
        all_global_accs.append(global_test_results['accu'])
        log('Starting fine-tuning phase')
        trial_local_final_accs = [] 
        trial_univ_local_accs = []   
        for client_idx in range(args.num_users):
            X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
            X_client_test, y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = False)
            for fine_tune_epoch in range(args.fine_tune_epochs):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                clients[client_idx], loss_dict = fine_tune_train(args, client_idx, clients[client_idx], X_train, y_train, 
                    is_train=True, body_train=False, coefs=coefs, log=print)
                clients[client_idx] = clients[client_idx].module
                ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
                
                # Log to TensorBoard
                writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
                writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
                log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')
                log(loss_dict)
            if epoch == num_train_epochs -1:
                client_test_results_global_data = local_test_global_model(args, clients[client_idx], X_test, y_test, coefs)
                trial_local_final_accs.append(ft_test_results['accu'])
                trial_univ_local_accs.append(client_test_results_global_data['accu'])                
        all_univ_local_accs.append(trial_univ_local_accs)
        all_local_accs.append(trial_local_final_accs)

        final_model_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(net_glob.state_dict(), final_model_path)
        log(f'Final model saved at {final_model_path}')
        
        final_client_model_path = os.path.join(model_dir, 'client_model.pth')
        torch.save(clients[0].state_dict(), final_client_model_path)
        log(f'Final model saved at {final_client_model_path}')
        
        net_glob = torch.nn.DataParallel(net_glob)
        clients[0] = torch.nn.DataParallel(clients[0])
        conduct_prototype_push(
        args=args,
            model=net_glob,
            dataset=dataset,
            server_idx=server_idx,
            dict_users=dict_users,
            device=device,
            epoch='final',
            model_dir=model_dir
        )
        conduct_prototype_push(
            args=args,
            model=clients[0],
            dataset=dataset,
            server_idx=server_idx,
            dict_users=dict_users,
            device=device,
            epoch='final',
            model_dir=os.path.join(model_dir,'clients'))
        
        logclose()
    gm_mean = np.mean(all_global_accs) * 100
    gm_std = np.std(all_global_accs) * 100

    trial_means = [np.mean(seed_accs) for seed_accs in all_local_accs]
    pm_mean = np.mean(trial_means) * 100
    pm_std = np.std(trial_means) * 100

    trial_means2 = [np.mean(seed_accs) for seed_accs in all_univ_local_accs]
    pm_univ_mean = np.mean(trial_means2) * 100
    pm_univ_std = np.std(trial_means2) * 100

    trial_means3 = [np.mean(seed_accs) for seed_accs in all_local_accs_bef_FT]
    pm_mean_bf_FT = np.mean(trial_means3) * 100
    pm_std_bf_FT = np.std(trial_means3) * 100
    results_to_save = {
        "global_model": {
            "mean": float(gm_mean),
            "std": float(gm_std),
            "raw_accs": [float(x) for x in all_global_accs]
        },
        "personalized_univ": {
            "mean": float(pm_univ_mean),
            "std": float(pm_univ_std),
            "raw_accs": [float(x) for x in trial_means2]
        },
        "personalized_model": {
            "mean": float(pm_mean),
            "std": float(pm_std),
            "trial_means": [float(x) for x in trial_means]
        },
        "personalized_model_bf_FT": {
            "mean": float(pm_mean_bf_FT),
            "std": float(pm_std_bf_FT),
            "trial_means": [float(x) for x in trial_means3]
        },
        "settings": {
            "base_seed": int(initial_seed),
            "num_users": int(args.num_users),
            "alpha": float(args.alpha) if hasattr(args, 'alpha') else None
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
    print(f"Global Model Accuracy (GM): {gm_mean:.2f}% ± {gm_std:.2f}")
    print(f"Personalized Model Accuracy (PM): {pm_mean:.2f}% ± {pm_std:.2f}")
    print("="*30)
if __name__ == "__main__":
    main()