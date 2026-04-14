# -*- coding: utf-8 -*-
import os
import shutil
import json
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import gc
import push
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
#from train_and_test_with_cluster_cost import *
from train_and_test_lr_schedule import *
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from utils.sampling import *
from utils.utils import *
#from Stiefel_model import *
from Gr_model_with_cluster_cost import *
from utils.misc import *


def _stream_global_test_results(args, model, dataset, server_idx, dict_users, coefs, eval_fn):
    weighted_batch_metrics = {
        'cross_entropy': 0.0,
        'cluster_loss': 0.0,
        'subspace_sep_loss': 0.0,
        'separation_loss': 0.0,
        'l1_loss': 0.0,
        'average loss': 0.0,
        'diversity loss': 0.0,
    }
    total_examples = 0
    total_batches = 0
    total_correct = 0.0

    for client_idx in range(args.num_users):
        X_client_test, y_client_test = load_Stan_data(
            args, dataset, server_idx, client_idx, dict_users, train=False
        )
        n_examples = len(y_client_test)
        if n_examples == 0:
            continue

        result = eval_fn(args, model, X_client_test, y_client_test, coefs)
        n_batches = max(1, (n_examples + args.local_bs - 1) // args.local_bs)

        total_examples += n_examples
        total_batches += n_batches
        total_correct += result['accu'] * n_examples

        for key in weighted_batch_metrics:
            weighted_batch_metrics[key] += result[key] * n_batches

        del X_client_test, y_client_test
        gc.collect()
        torch.cuda.empty_cache()

    if total_examples == 0 or total_batches == 0:
        raise RuntimeError('Global test split is empty.')

    aggregated = {
        key: weighted_batch_metrics[key] / total_batches
        for key in weighted_batch_metrics
    }
    aggregated['accu'] = total_correct / total_examples
    aggregated['l1'] = model.last_layer.weight.norm(p=1).item()
    return aggregated


def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    coefs = settings_CUB.coefs
    num_train_epochs = args.num_train_epochs
    #user_datasets= load_cropped_CUB_random_distribution(num_users=args.num_users, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True)
    prototype_per_class = settings_CUB.prototype_per_class
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
        model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '/'+ str(args.iid) + '/'+ str(start_time) + '/' + 'FedTesBABU NonCayleyT'
        if os.path.exists(model_dir) is True:
            shutil.rmtree(model_dir)
        makedir(model_dir)
        log_directory = f"checkpoints/{str(args.iid)}/'FedTesBABU NonCayleyT'/{start_time+'tuning with interval without augmented dataset'}/"
        writer = SummaryWriter(log_dir=log_directory)    
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
        dict_users, server_idx, dataset=setup_datasets(args)
        net_glob = construct_TesNet(args= args, base_architecture=base_architecture, prototype_per_class=prototype_per_class, dataset=dataset_name,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_shape,
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
        
        net_glob = net_glob.to(device)
        net_glob.prototype_vectors = net_glob.prototype_vectors.to(device)
        
        # Main training loop
        for epoch in range(num_train_epochs):
            log('epoch: \t{0}'.format(epoch))
            #return_dict = {}
            clients_state_list = []
            args.epoch = epoch
            # Train each client
            for client_idx in range(args.num_users):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                #X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
                X_train,y_train=load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= True)
                #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
                clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train, 
                                                    is_train=True, body_train=True, coefs=coefs, log=log)
                #return_dict[client_idx] = loss_dict
                clients_state_list.append(clients[client_idx].module.state_dict())
                clients[client_idx] = clients[client_idx].module
                X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                log(f'Client {client_idx} test accuracy before aggregation: {client_test_results["accu"]}')
                #log(f'Client {client_idx} loss values before aggregation: {loss_dict}')

            net_glob.load_state_dict(FedAvg(clients_state_list, net_glob.state_dict()))
            
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
            glob_state = net_glob.state_dict()
            for user_idx in range(args.num_users):
                for k in update_keys:
                    clients_state_list[user_idx][k] = copy.deepcopy(glob_state[k])
                clients[user_idx].load_state_dict(clients_state_list[user_idx])
            del glob_state
            del clients_state_list
            gc.collect()
            torch.cuda.empty_cache()

            global_test_results = _stream_global_test_results(args, net_glob, dataset, server_idx, dict_users, coefs, local_test_global_model_proto)
            writer.add_scalar('Global/Loss', global_test_results['average loss'], global_step=epoch)
            writer.add_scalar('Global/Accuracy', global_test_results['accu'], global_step=epoch) 
            log('Global model test')
            log(global_test_results)
            
            # Test individual clients on their test data
            for client_idx in range(args.num_users):
                #X_train, y_train, X_client_test, y_client_test = load_data_random(user_datasets, client_idx)
                #X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
                X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                client_test_results_global_data = _stream_global_test_results(args, clients[client_idx], dataset, server_idx, dict_users, coefs, local_test_global_model_proto)
                writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
                writer.add_scalar(f'Client_{client_idx}/Accuracy_glob_data', client_test_results_global_data['accu'], global_step=epoch)
                log(f'Client {client_idx} test accuracy after aggregating global model: {client_test_results["accu"]}')
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
                    is_train=True, body_train=False, coefs=coefs, log=log)
                clients[client_idx] = clients[client_idx].module
                ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
                
                # Log to TensorBoard
                writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
                writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
                log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')
                #log(loss_dict)
                if epoch == num_train_epochs -1:
                    client_test_results_global_data = _stream_global_test_results(args, clients[client_idx], dataset, server_idx, dict_users, coefs, local_test_global_model)
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
        
        # Final prototype push
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

        writer.close()
        del dataset, dict_users, server_idx, clients, net_glob
        gc.collect()
        torch.cuda.empty_cache()
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