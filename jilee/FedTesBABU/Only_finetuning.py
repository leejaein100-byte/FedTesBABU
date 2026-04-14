from torch.utils.tensorboard import SummaryWriter
from utils.misc import *
from utils.sampling import *
from utils.utils import *
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import *
from util.log import create_logger
import time
import settings_CUB
from Gr_model import *
from train_and_test_Gr import *

start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
model_dir = "/home/jilee/jilee/FedTesBABU/saved_models/Stanford_dog/resnet50/False/2025-11-04_19-53-20/FedTesBABU NonCayleyT"
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'fine_tune_train.log'))
args = settings_CUB.args
num_classes = settings_CUB.num_classes
img_size = args.img_size
add_on_layers_type = settings_CUB.add_on_layers_type
base_architecture = args.arch
prototype_shape = settings_CUB.prototype_shape
prototype_activation_function = settings_CUB.prototype_activation_function
coefs = settings_CUB.coefs
dataset_name = args.dataset#user_datasets= load_cropped_CUB_random_distribution(num_users=args.num_users, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True)
prototype_per_class = settings_CUB.prototype_per_class
dict_users, server_idx, dataset=setup_datasets(args)
log_directory = f"checkpoints/{str(args.iid)}/'FedTesBABU NonCayleyT_FTonly'/{start_time+'tuning with interval without augmented dataset'}/"
writer = SummaryWriter(log_dir=log_directory)
load_model_path = os.path.join(model_dir,"final_model.pth")
print('load model path', load_model_path)
net_glob = construct_TesNet(
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
update_keys = [k for k in net_glob.state_dict().keys() 
                    if 'linear' not in k and 'last_layer' not in k]
if os.path.exists(load_model_path):
    log(f"Loading pretrained model from: {load_model_path}")
    state_dict = torch.load(load_model_path, map_location='cpu')
    net_glob.load_state_dict(state_dict)
    log("Model state loaded successfully.")
else:
    log("No pretrained model path specified, initializing new model.")

clients = []
for _ in range(args.num_users):
    clients.append(copy.deepcopy(net_glob))
clients_state_list = []
#for user_idx in range(args.num_users):
#    for k in update_keys:
#        clients_state_list[user_idx][k] = copy.deepcopy(net_glob.state_dict()[k])    
#    clients[user_idx].load_state_dict(clients_state_list[user_idx])
log('Starting fine-tuning phase')
client_ft_results = {}
for client_idx in range(args.num_users):
    client_ft_results[client_idx] = []
    X_train, y_train = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = True, private = True)
    X_client_test, y_client_test = load_Stan_data(args, dataset, server_idx, client_idx, dict_users, train = False)
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

    #final_model_path = os.path.join(model_dir, 'final_model.pth')
    #torch.save(net_glob.state_dict(), final_model_path)
    #log(f'Final model saved at {final_model_path}')
    
    final_client_model_path = os.path.join(model_dir, 'client_model.pth')
    torch.save(clients[0].state_dict(), final_client_model_path)
    log(f'Final model saved at {final_client_model_path}')

logclose()