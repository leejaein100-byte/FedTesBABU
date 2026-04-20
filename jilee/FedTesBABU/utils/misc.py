from utils.sampling import *
from utils.utils import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from util.log import create_logger
import push
import json
import settings_CUB
from Gr_model_with_cluster_cost import *
#from util.preprocess import preprocess_input_function

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

def conduct_prototype_push_lazy(args, model, lazy_dataset, dict_users, device, epoch, model_dir):
    """Memory-efficient prototype push that streams training data per-client instead of
    loading the entire dataset into RAM at once."""
    from torch.utils.data import Dataset as _Dataset

    dict_users_train, _ = dict_users
    all_indices = []
    for uid in range(args.num_users):
        all_indices.extend(dict_users_train.get(uid, []))

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    class _DenormSubset(_Dataset):
        def __init__(self, base_dataset, indices):
            self.base = base_dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, label = self.base[self.indices[i]]
            return img * std_t + mean_t, label

    push_dataset = _DenormSubset(lazy_dataset, all_indices)
    push_dataloader = torch.utils.data.DataLoader(
        push_dataset, batch_size=args.local_bs, shuffle=False,
        num_workers=4, pin_memory=False)

    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)

    model = model.to(device)
    model_parallel = model.module if hasattr(model, 'module') else model

    push.push_prototypes_top3(
        dataloader=push_dataloader,
        prototype_network_parallel=model_parallel,
        class_specific=True,
        preprocess_input_function=None,
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=img_dir,
        epoch_number=epoch,
        prototype_img_filename_prefix='prototype-img',
        prototype_self_act_filename_prefix='prototype-self-act',
        proto_bound_boxes_filename_prefix='bb',
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
    'arch': args.arch,
    'num_channels': args.num_channels,
    'server_id_size': args.server_id_size,
    'local_bs': args.local_bs,
    'num_users': args.num_users,
    'dataset': args.dataset,
    'SL_epochs': args.SL_epochs,
    'fine_tune_epochs': args.fine_tune_epochs,
    'alpha': args.alpha,
    'use_bbox': args.use_bbox,
    'temp': args.temp,
    'kd_epochs':args.kd_epochs,
    'use_fisher':args.use_fisher,
    'ewc_lambda':args.ewc_lambda,
    'reg_lambda':getattr(args, 'reg_lambda', 0.0),
    'min_per_label':args.min_per_label,
    'hyperparam': args.hyperparam,
    'num_teachers': args.num_teachers,
    'patch_num': args.patch_num,
    'score logits':args.score_logit,
    'last layer':args.last_layer,
    'cons_mode': args.cons_mode,
    'reg_lambda_eucl': args.reg_lambda_eucl,
    'reg_lmabda_proj':args.reg_lambda_proj
    }

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