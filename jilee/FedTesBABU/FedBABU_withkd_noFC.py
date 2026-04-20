import torch.nn.functional as F
import numpy as np
from scipy.stats import mode
import os
import shutil
import json
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
from train_and_test_with_cluster_cost import *
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from utils.utils import *
from Gr_model_with_cluster_cost import *
from utils.misc import *

def enhanced_training_with_bayesian_kd_server(args, clients, clients_state_list, net_glob, dict_users, server_idx, dataset, 
                                            coefs, device, log, num_classes, epoch, prototype_per_class):
                                           
    """
    Enhanced training function where net_glob is trained with server data using KD from Dirichlet teachers
    """
    
    # Initialize Bayesian KD trainer with net_glob as the main object
    kd_trainer = BayesianKDTrainer(
        args, num_classes, net_glob, prototype_per_class,
        alpha=0.5, temp=args.temp, hyperparam=args.hyperparam,
        reg_lambda=getattr(args, 'reg_lambda', 0.0),
    )
    # Create teacher models using Dirichlet distribution
    teacher_models = kd_trainer.create_teacher_models_with_dirichlet(
        clients_state_list, clients, device, num_teachers=args.num_teachers,
    )
    # Load server data once (moved outside the kd_epochs loop)
    X_server, y_server = load_Stan_data(args, dataset, server_idx, 0, dict_users, private=False)
    server_data = (X_server, y_server)

    # Save FedAvg anchor BEFORE any KD update.
    # Uses plain L2 for Euclidean params and projection metric for prototype_vectors.
    if getattr(args, 'reg_lambda', 0.0) > 0.0:
        kd_trainer.save_anchor(device=device)

    for i in range(args.kd_epochs):
        global_kd_results = kd_trainer.train_global_model_with_kd(
            teacher_models, server_data, device)

        #log(f'Global KD training - Total Loss: {global_kd_results["average loss"]:.4f}')
    
    
    # Phase 5: Update clients by copying parameters from trained net_glob
    #log(f'Epoch {epoch}: Updating clients from trained global model')
    #clients = kd_trainer.update_clients_from_global(
    #    clients, update_keys
    #)
    
    log(f'Epoch {epoch}: Bayesian KD training completed with server-based global training')
    log(f'Global KD training results: {global_kd_results}')
    
    #return global_kd_results, clients
    return  global_kd_results, kd_trainer.net_glob

def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '/' + str(args.iid) + '/'+str(start_time) + '/' + 'FedTesBABU_with_KD_Noncayley'

    if os.path.exists(model_dir) is True:
        shutil.rmtree(model_dir)
    makedir(model_dir)
    #backup_dir = f"checkpoints2/{str(args.iid)}/{str(args.iid)}/'FedTesBABU_with_KD'/{start_time+'without augmented dataset'}/"
    log_directory = f"checkpoints/{str(args.iid)}/'FedTesBABU_with_KD_NonCayley'/{start_time+'without augmented dataset'}/"
    writer = SummaryWriter(log_dir=log_directory)    
    shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
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
    dict_users, server_idx, dataset=setup_datasets(args)
    net_glob = construct_TesNet(args= args, base_architecture=base_architecture, prototype_per_class=prototype_per_class, dataset=dataset_name,
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
        X_temp_test,y_temp_test= load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
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
        
        for client_idx in range(args.num_users):
            clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
            #X_train, y_train = load_data(args, dataset, server_id, client_idx, dict_users_train, train = True, private = True)
            X_train,y_train=load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= True)
            #X_train, y_train, X_temp_test, y_temp_test = load_data_random(user_datasets, client_idx)
            clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train, 
                                                 is_train=True, coefs=coefs, log=log)
                                                 
            #return_dict[client_idx] = loss_dict
            clients_state_list.append(clients[client_idx].module.state_dict())
            clients[client_idx] = clients[client_idx].module
            X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
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

        # Test global model
        global_test_results = local_test_global_model_proto(args, net_glob, X_test, y_test, coefs)  
        writer.add_scalar('Global/Loss', global_test_results['average loss'], global_step=epoch)
        writer.add_scalar('Global/Accuracy', global_test_results['accu'], global_step=epoch) 
        log('Global model test')
        log(global_test_results)
        
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
            X_client_test,y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train= False)
            client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
            client_test_results_global_data = local_test_global_model_proto(args, clients[client_idx], X_test, y_test, coefs)
            writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
            writer.add_scalar(f'Client_{client_idx}/Accuracy_glob_data', client_test_results_global_data['accu'], global_step=epoch)
            log(f'Client {client_idx} test accuracy after aggregation global data: {client_test_results_global_data["accu"]}')
            log(f'Client {client_idx} test accuracy after aggregation client data: {client_test_results["accu"]}')

    #log('Starting fine-tuning phase')
    #client_ft_results = {}
    #for client_idx in range(args.num_users):
    #    client_ft_results[client_idx] = []
    #    X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
    #    X_client_test, y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = False, private = True)
    #    for fine_tune_epoch in range(args.fine_tune_epochs):
    #        clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
    #        clients[client_idx], loss_dict = fine_tune_train(args, client_idx, clients[client_idx], X_train, y_train, 
    #            is_train=True, coefs=coefs, log=print)
    #        clients[client_idx] = clients[client_idx].module
    #        ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)
    #       client_ft_results[client_idx].append(ft_test_results['accu'])
            
            # Log to TensorBoard
    #        writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
    #        writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
    #        log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')
    #        log(loss_dict)     

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
if __name__ == "__main__":
    main()