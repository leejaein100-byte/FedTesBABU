import torch.nn.functional as F
import numpy as np
from scipy.stats import mode
import os
import shutil
import json
import copy
import gc
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torch.utils.tensorboard import SummaryWriter
from util.helpers import makedir
#from train_and_test_with_cluster_cost import *
from train_and_test_lr_schedule_simul import *
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import settings_CUB
#from utils.Fedtes_data_dist_img_augment import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from utils.utils import *
from Gr_model_with_cluster_cost import *
from utils.misc import *


def _stream_global_test_results(args, model, lazy_dataset, dict_users, coefs, eval_fn, test_cache=None):
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
        if test_cache is not None and client_idx in test_cache:
            X_client_test, y_client_test = test_cache[client_idx]
        else:
            X_client_test, y_client_test = load_Stan_data_lazy(
                lazy_dataset, dict_users, None, client_idx, train=False
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

        if test_cache is None or client_idx not in test_cache:
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


def enhanced_training_with_bayesian_kd_server(args, clients, clients_state_list, net_glob, dict_users, server_idx, lazy_dataset,
                                            coefs, device, log, num_classes, epoch, prototype_per_class):

    """
    Enhanced training function where net_glob is trained with server data using KD from Dirichlet teachers
    """

    # Initialize Bayesian KD trainer with net_glob as the main object
    kd_trainer = BayesianKDTrainer(
        args, num_classes, net_glob, prototype_per_class,
        alpha=0.5, temp=args.temp, hyperparam=args.hyperparam,
        ewc_lambda=getattr(args, 'ewc_lambda', 0.0),
        use_fisher=getattr(args, 'use_fisher', False),
        reg_lambda_eucl=getattr(args, 'reg_lambda_eucl', 0.0),
        reg_lambda_proj=getattr(args, 'reg_lambda_proj', 0.0),
    )
    # Phase 2: Create teacher models using Dirichlet distribution
    teacher_models = kd_trainer.create_teacher_models_with_dirichlet(
        clients_state_list, clients, device, num_teachers=args.num_teachers,
    )
    # Load server data once — reused for anchor/Fisher and KD training
    X_server, y_server = load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, 0, private=False)
    server_data = (X_server, y_server)

    # Save FedAvg anchor BEFORE any KD update.
    # ewc_lambda>0: Fisher estimated on 256-sample subsample (or plain L2 if use_fisher=False).
    # reg_lambda>0: geometry-aware anchor (L2 for Euclidean, projection metric for prototypes).
    # Both share the same anchor_state snapshot; only one save_anchor() call is needed.
    if (getattr(args, 'ewc_lambda', 0.0) > 0.0
            or getattr(args, 'reg_lambda_eucl', 0.0) > 0.0
            or getattr(args, 'reg_lambda_proj', 0.0) > 0.0):
        kd_trainer.save_anchor(server_data=server_data, device=device, n_fisher_samples=256)

    # Train net_glob with knowledge distillation; optimizer persists across kd_epochs
    global_kd_results = kd_trainer.train_global_model_with_kd(
        teacher_models, server_data, device, kd_epochs=args.kd_epochs)

    del X_server, y_server
    gc.collect()

    log(f'Epoch {epoch}: Bayesian KD training completed with server-based global training')
    log(f'Global KD training results: {global_kd_results}')

    return  global_kd_results, kd_trainer.net_glob

def main():
    args = settings_CUB.args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Device_id)
    dataset_name = args.dataset
    base_architecture = args.arch
    initial_seed = args.seed
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    num_classes = settings_CUB.num_classes
    img_size = args.img_size
    add_on_layers_type = settings_CUB.add_on_layers_type
    prototype_shape = settings_CUB.prototype_shape
    prototype_activation_function = settings_CUB.prototype_activation_function
    coefs = settings_CUB.coefs
    num_train_epochs = args.num_train_epochs
    prototype_per_class = settings_CUB.prototype_per_class
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
        model_dir = './saved_models/' + dataset_name+'/' + base_architecture + '/' + str(args.iid) + '/'+str(start_time) + '/' + 'FedTesBABU_with_KD_Noncayley'

        if os.path.exists(model_dir) is True:
            shutil.rmtree(model_dir)
        makedir(model_dir)
        log_directory = f"checkpoints/{str(args.iid)}/{str(dataset_name)}/'FedTesBABU_with_KD_NonCayley'/{start_time+'without augmented dataset'}/"
        writer = SummaryWriter(log_dir=log_directory)
        shutil.copy(src=os.path.join(os.getcwd(), 'settings_CUB.py'), dst=model_dir)
        dict_users, server_idx, lazy_dataset = setup_datasets_lazy(args)
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

    # Main training loop with Bayesian Knowledge Distillation
        for epoch in range(num_train_epochs):
            log('epoch: \t{0}'.format(epoch))
            #return_dict = {}
            clients_state_list = []
            args.epoch = epoch
            test_cache = {}
            for client_idx in range(args.num_users):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                X_train, y_train = load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, client_idx, True)
                clients[client_idx], loss_dict = train(args, client_idx, clients[client_idx], X_train, y_train,
                                                    is_train=True, coefs=coefs, log=log)
                #print('client_idx', client_idx)
                #return_dict[client_idx] = loss_dict
                clients_state_list.append(clients[client_idx].module.state_dict())
                clients[client_idx] = clients[client_idx].module
                del X_train, y_train 
                gc.collect()
                X_client_test, y_client_test = load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, client_idx, train=False)
                test_cache[client_idx] = (X_client_test, y_client_test)
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                log(f'Client {client_idx} test accuracy before aggregation: {client_test_results["accu"]}')
                gc.collect()

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
                temp1 = consensus_update(Gm_manifold[cls], mode = args.cons_mode)
                temp2.append(torch.unsqueeze(temp1, 0))

            print('frechet mean finished')
            net_glob.prototype_vectors.data = torch.cat(temp2, dim=0)
            del temp1, temp2

            if epoch >= args.warmup_ep:
                kd_return_dict, net_glob = enhanced_training_with_bayesian_kd_server(
                    args, clients, clients_state_list, net_glob, dict_users, server_idx, lazy_dataset,
                    coefs, device, log, num_classes, epoch, prototype_per_class)
                # Update clients with global model parameters
            for user_idx in range(args.num_users):
                for k in update_keys:
                    clients_state_list[user_idx][k] = copy.deepcopy(net_glob.state_dict()[k])
                clients[user_idx].load_state_dict(clients_state_list[user_idx])

            for client_idx in range(args.num_users):
                X_client_test, y_client_test = test_cache[client_idx]
                client_test_results = local_test_global_model_proto(args, clients[client_idx], X_client_test, y_client_test, coefs)
                client_test_results_global_data = _stream_global_test_results(args, clients[client_idx], lazy_dataset, dict_users, coefs, local_test_global_model_proto, test_cache=test_cache)
                writer.add_scalar(f'Client_{client_idx}/Accuracy', client_test_results['accu'], global_step=epoch)
                writer.add_scalar(f'Client_{client_idx}/Accuracy_glob_data', client_test_results_global_data['accu'], global_step=epoch)
                log(f'Client {client_idx} test accuracy after aggregation global data: {client_test_results_global_data["accu"]}')
                log(f'Client {client_idx} test accuracy after aggregation client data: {client_test_results["accu"]}')
                if epoch == num_train_epochs -1:
                    all_local_accs_bef_FT.append(client_test_results['accu'])

            global_test_results = _stream_global_test_results(args, net_glob, lazy_dataset, dict_users, coefs, local_test_global_model_proto, test_cache=test_cache)
            writer.add_scalar('Global/Loss', global_test_results['average loss'], global_step=epoch)
            writer.add_scalar('Global/Accuracy', global_test_results['accu'], global_step=epoch)
            log('Global model test')
            log(global_test_results)

            test_cache.clear()
            del test_cache, clients_state_list
            gc.collect()
            torch.cuda.empty_cache()

        all_global_accs.append(global_test_results['accu'])

        log('Starting fine-tuning phase')
        trial_local_final_accs = []
        trial_univ_local_accs = []
        # Pre-load all clients' test data once; reused in every _stream_global_test_results call
        log('Building fine-tuning test cache...')
        ft_test_cache = {i: load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, i, train=False)
                         for i in range(args.num_users)}
        for client_idx in range(args.num_users):
            X_train, y_train = load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, client_idx, True)
            X_client_test, y_client_test = ft_test_cache[client_idx]
            for fine_tune_epoch in range(args.fine_tune_epochs):
                clients[client_idx] = torch.nn.DataParallel(clients[client_idx])
                clients[client_idx], loss_dict = fine_tune_train(args, client_idx, clients[client_idx], X_train, y_train,
                    is_train=True, coefs=coefs, log=log)
                clients[client_idx] = clients[client_idx].module
                ft_test_results = local_test_global_model(args, clients[client_idx], X_client_test, y_client_test, coefs)

                # Log to TensorBoard
                writer.add_scalar(f'FineTune/Client_{client_idx}_Accuracy', ft_test_results['accu'], global_step=fine_tune_epoch)
                writer.add_scalar(f'FineTune/Client_{client_idx}_loss', loss_dict['average loss'], global_step=fine_tune_epoch)
                log(f'Client {client_idx}, Fine-tune epoch {fine_tune_epoch}, accuracy: {ft_test_results["accu"]}')
                log(loss_dict)
            client_test_results_global_data = _stream_global_test_results(args, clients[client_idx], lazy_dataset, dict_users, coefs, local_test_global_model, test_cache=ft_test_cache)
            trial_local_final_accs.append(ft_test_results['accu'])
            trial_univ_local_accs.append(client_test_results_global_data['accu'])
            del X_train, y_train; gc.collect()
        ft_test_cache.clear()
        del ft_test_cache

        all_univ_local_accs.append(trial_univ_local_accs)
        all_local_accs.append(trial_local_final_accs)
        # Create a visualization of client accuracies after fine-tuning
        final_model_path = os.path.join(model_dir, 'final_model.pth')
        torch.save(net_glob.state_dict(), final_model_path)
        log(f'Final model saved at {final_model_path}')

        final_client_model_path = os.path.join(model_dir, 'client_model.pth')
        torch.save(clients[0].state_dict(), final_client_model_path)
        log(f'Final model saved at {final_client_model_path}')

        conduct_prototype_push_lazy(
            args=args,
            model=net_glob,
            lazy_dataset=lazy_dataset,
            dict_users=dict_users,
            device=device,
            epoch='final',
            model_dir=model_dir
        )
        conduct_prototype_push_lazy(
            args=args,
            model=clients[0],
            lazy_dataset=lazy_dataset,
            dict_users=dict_users,
            device=device,
            epoch='final',
            model_dir=os.path.join(model_dir, 'clients'))

        writer.close()
        del lazy_dataset, dict_users, server_idx, clients, net_glob
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
