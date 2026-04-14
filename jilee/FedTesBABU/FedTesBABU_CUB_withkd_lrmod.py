import torch.nn.functional as F
import numpy as np
import os
import shutil
import json
import copy
import time
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
from train_and_test_lr_schedule import *
from util.log import create_logger
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Fedtes_data_dist import *
from utils.utils import *
from Gr_model_with_cluster_cost import *
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
        #X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train=True)
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
    'seed': args.seed,
    'iid': args.iid,
    'server_id_size': args.server_id_size,
    'local_bs': args.local_bs,
    'num_users': args.num_users,
    'arch': args.arch,
    'dataset': args.dataset,
    'SL_epochs': args.SL_epochs,
    'fine_tune_epochs': args.fine_tune_epochs,
    'alpha': args.alpha,
    'temp': args.temp,
    'warmup_ep':args.warmup_ep,
    'hyperparam': args.hyperparam,
    'num_teachers': args.num_teachers,
    'patch_num': args.patch_num,
    'score logits':args.score_logit,
    'Tscale': args.Tscale,
    'kd_epochs': args.kd_epochs
    }

    # Save settings to JSON file
    settings_file = os.path.join(settings_dir, 'settings.json')
    with open(settings_file, 'w') as f:
        json.dump(settings_dict, f, indent=4)

def load_model_and_push(model_path, args):
    """
    Load a trained model from the specified path and conduct prototype pushing.
    """
    device = torch.device(f'cuda:0')
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

def enhanced_training_with_bayesian_kd_server(args, clients, clients_state_list, net_glob, dict_users, server_idx, dataset, 
                                            coefs, device, log, num_classes, epoch, prototype_per_class):
                                           
    """
    Enhanced training function where net_glob is trained with server data using KD from Dirichlet teachers
    """
    
    # Initialize Bayesian KD trainer with net_glob as the main object
    #kd_trainer = BayesianKDTrainer(args, num_classes, net_glob, prototype_per_class, alpha=2.0, temp=4.0, kd_weight=0.7)
    kd_trainer = BayesianKDTrainer(args, num_classes, net_glob, prototype_per_class, alpha=5.0, temp=args.temp, hyperparam = args.hyperparam)
    # Phase 2: Create teacher models using Dirichlet distribution
    #log(f'Epoch {epoch}: Creating teacher models with Dirichlet distribution')
    teacher_models = kd_trainer.create_teacher_models_with_dirichlet(
        clients_state_list, clients, device, num_teachers=args.num_teachers,
    )
    # Phase 1: Standard federated training on clients
    for i in range(args.kd_epochs):
        # Use server_idx as a placeholder - get server data 
        X_server, y_server = load_data(args, dataset, server_idx, 0, dict_users, private=False)
        server_data = (X_server, y_server)
        
        # Phase 4: Train net_glob with knowledge distillation using server data
        #log(f'Epoch {epoch}: Training global model with knowledge distillation on server data')
        global_kd_results = kd_trainer.train_global_model_with_kd(
            teacher_models, server_data, device)
    
    log(f'Epoch {epoch}: Bayesian KD training completed with server-based global training')
    log(f'Global KD training results: {global_kd_results}')
    
    return  global_kd_results, kd_trainer.net_glob

def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch
    
    #shutil.copytree(log_directory, backup_dir, dirs_exist_ok=True)
    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    coefs = settings_CUB.coefs
    num_train_epochs = args.num_train_epochs
    args.ngpus_per_node = torch.cuda.device_count()
    #user_datasets= load_cropped_CUB_random_distribution(num_users=args.num_users, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True)
    prototype_per_class = settings_CUB.prototype_per_class
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    all_global_accs = [] # [seed1_acc, seed2_acc, seed3_acc]
    all_local_accs = []  # [ [c1_acc, c2_acc...], [c1_acc...], [c1_acc...] ]
    all_local_accs_bef_FT = []
    all_univ_local_accs = []
    device = torch.device(f'cuda:0')
    initial_seed = args.seed
    
    for trial in range(3):
        current_seed = initial_seed + trial
        args.seed = current_seed
        print(f"\n === Starting Trial {trial+1}/3 with Seed: {current_seed} === \n")
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '/' + str(args.iid) + '/'+str(start_time) + '/' + 'FedTesBABU_with_KD_Noncayley_lr_schedule'
        if os.path.exists(model_dir) is True:
            shutil.rmtree(model_dir)
        makedir(model_dir)
        log_directory = f"checkpoints/{str(args.iid)}/{str(dataset_name)}/'FedTesBABU_with_KD_NonCayley_lrschedule'/{start_time+'without augmented dataset'}/"
        writer = SummaryWriter(log_dir=log_directory)    
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
        dict_users, server_idx, dataset=setup_datasets(args)
        net_glob = construct_TesNet(args= args, base_architecture=base_architecture, prototype_per_class=prototype_per_class, dataset=dataset_name,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
        save_settings(args, model_dir)

        update_keys = [k for k in net_glob.state_dict().keys() 
                      if 'linear' not in k and 'last_layer' not in k]
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
     
        clients = []
        for _ in range(args.num_users):
            clients.append(copy.deepcopy(net_glob))
        
        # Prepare combined test data
        X_test = []
        y_test = []
        for client_idx in range(args.num_users):
            #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
            X_temp_test, y_temp_test = load_data(args, dataset, server_idx, client_idx, dict_users, train = False, private = True)
            X_test.append(X_temp_test)
            y_test.append(y_temp_test)
        
        X_test = torch.cat(X_test)
        y_test = torch.cat(y_test)
        net_glob = net_glob.to(device)
        net_glob.prototype_vectors = net_glob.prototype_vectors.to(device)

    # Main training loop with Bayesian Knowledge Distillation
        for epoch in range(num_train_epochs):
            log('epoch: \t{0}'.format(epoch))
            #return_dict = {}
            clients_state_list = []
            args.epoch = epoch
            for client_idx in range(args.num_users):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                X_train, y_train = load_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
                #X_train,y_train=load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= True)
                #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
                clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train, 
                                                    is_train=True, coefs=coefs, log=log)
                clients_state_list.append(clients[client_idx].module.state_dict())
                clients[client_idx] = clients[client_idx].module
                X_client_test, y_client_test = load_data(args, dataset, server_idx, client_idx, dict_users, train = False, private = True)
                #X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                log(f'Client {client_idx} test accuracy before aggregation: {client_test_results["accu"]}')
                #log(f'Client {client_idx} loss values before aggregation: {loss_dict}')        

            net_glob.load_state_dict(FedAvg(clients_state_list, net_glob.state_dict()))
            
            # Consensus update for prototype vectors
            temp2 = []
            Gm_manifold = []
            
            # Collect prototype vectors from all clients and global model
            for client_idx in range(args.num_users):
                Gm_manifold.append(torch.unsqueeze(clients[client_idx].prototype_vectors.data.to(device), 0).to(device))

            Gm_manifold = torch.cat(Gm_manifold, dim=0)        
            if Gm_manifold.dim() == 4:
                Gm_manifold = Gm_manifold.permute(1, 0, 2, 3)
            else:
                print(f"Warning: Gm_manifold has unexpected shape: {Gm_manifold.shape}")
            
            # Compute Frechet mean for each class
            for cls in range(num_classes):
                temp1 = consensus_update(Gm_manifold[cls])
                temp2.append(torch.unsqueeze(temp1, 0))
            
            print('frechet mean finished')
            net_glob.prototype_vectors.data = torch.cat(temp2, dim=0)     
            del temp1, temp2

            if epoch >= args.warmup_ep:
                kd_return_dict, net_glob = enhanced_training_with_bayesian_kd_server(
                    args, clients, clients_state_list, net_glob, dict_users, server_idx, dataset, 
                    coefs, device, log, num_classes, epoch, prototype_per_class)
                #log('KD results')
                #log(kd_return_dict)
                # Update clients with global model parameters
            for user_idx in range(args.num_users):
                for k in update_keys:
                    clients_state_list[user_idx][k] = copy.deepcopy(net_glob.state_dict()[k])    
                clients[user_idx].load_state_dict(clients_state_list[user_idx])

            for client_idx in range(args.num_users):
                #X_train, y_train, X_client_test, y_client_test = load_data_random(user_datasets, client_idx)
                X_client_test, y_client_test = load_data(args, dataset, server_idx, client_idx, dict_users, train = False, private = True)
                #X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                client_test_results_global_data = local_test_global_model_proto(args, clients[client_idx], X_test, y_test, coefs)
                writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
                writer.add_scalar(f'Client_{client_idx}/Accuracy_glob_data', client_test_results_global_data['accu'], global_step=epoch)
                log(f'Client {client_idx} test accuracy after aggregation global data: {client_test_results_global_data["accu"]}')
                log(f'Client {client_idx} test accuracy after aggregation client data: {client_test_results["accu"]}')
                if epoch == num_train_epochs -1:
                    all_local_accs_bef_FT.append(client_test_results['accu'])
            global_test_results = local_test_global_model_proto(args, net_glob, X_test, y_test, coefs)  
            writer.add_scalar('Global/Loss', global_test_results['average loss'], global_step=epoch)
            writer.add_scalar('Global/Accuracy', global_test_results['accu'], global_step=epoch) 
            log('Global model test')
            log(global_test_results)
        all_global_accs.append(global_test_results['accu'])
        
        log('Starting fine-tuning phase')
        trial_local_final_accs = [] 
        trial_univ_local_accs = [] 
        for client_idx in range(args.num_users):
            #client_ft_results[client_idx] = []
            X_train, y_train = load_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
            X_client_test, y_client_test = load_data(args, dataset, server_idx, client_idx, dict_users, train = False, private = True)
            for fine_tune_epoch in range(args.fine_tune_epochs):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                clients[client_idx], loss_dict = fine_tune_train(args, client_idx, clients[client_idx], X_train, y_train, 
                    is_train=True, coefs=coefs, log=log)
                clients[client_idx] = clients[client_idx].module
                ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)

                writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
                writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
                log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')   
            if epoch == num_train_epochs -1:
                client_test_results_global_data = local_test_global_model(args, clients[client_idx], X_test, y_test, coefs)
                trial_local_final_accs.append(ft_test_results['accu'])
                trial_univ_local_accs.append(client_test_results_global_data['accu'])
        all_univ_local_accs.append(trial_univ_local_accs)
        all_local_accs.append(trial_local_final_accs)
        # Create a visualization of client accuracies after fine-tuning
        final_model_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(net_glob.state_dict(), final_model_path)
        log(f'Final model saved at {final_model_path}')

        final_client_model_path = os.path.join(model_dir, 'client_model.pth')
        torch.save(clients[0].state_dict(), final_client_model_path)
        log(f'Final model saved at {final_client_model_path}')
        
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