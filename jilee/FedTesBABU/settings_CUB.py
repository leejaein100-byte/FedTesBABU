import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
#parser.add_argument('--num_gpu', type=int, default=1, help="GPU ID, -1 for CPU")
#parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--arch',type=str, default='resnet50')
parser.add_argument('--dataset',type=str, default='Stanford_dog')
parser.add_argument('--SL_epochs',type=int, default=5)
parser.add_argument('--warmup_ep',type=int, default=0)
parser.add_argument('--fine_tune_epochs',type=int, default=30)
parser.add_argument('--num_classes',type=int, default=120)
parser.add_argument('--patch_num',type=int, default=3, help='1 or 3')
parser.add_argument('--alpha',type=float, default=0.5)
parser.add_argument('--cons_mode',type=str, default='polar')
parser.add_argument('--use_bbox', type=bool, default=True,
                    help='Use bounding box cropping')
parser.add_argument('--Tscale', action='store_true', help='Enable Tscale') 
parser.add_argument('--load_model_path', type=str, default=None, 
                    help='Path to a pretrained model state_dict to load.')
parser.add_argument('--update_interval', type=int, default=10)
parser.add_argument('--num_channels',type=int, default=64)
parser.add_argument('--score_logit',action='store_true', help = 'whether using score as a final logit')
parser.add_argument('--last_layer',action='store_true', help = 'whether using score as a final logit')
parser.add_argument('--Device_id',type=str, default='3')
parser.add_argument('--img_size',type=int, default=224)
parser.add_argument('--iid', action='store_true', help="Use IID distribution")
parser.add_argument('--num_users', type=int, default=8, help="number of users")
parser.add_argument('--num_teachers', type=int, default=30, help="number of teachers")
parser.add_argument('--tr_frac', type=float, default=0.8, help='model name')
parser.add_argument('--temp', type=float, default=1, help='model name')
parser.add_argument('--hyperparam', type=float, default=0)
parser.add_argument('--local_bs', type=int, default=128, help="local batch size")
parser.add_argument('--server_id_size', type=int, default= 3328, help="server_id size")
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--num_train_epochs', type=int, default= 150)
parser.add_argument('--seed', type=int, default= 42, help="SEED")
parser.add_argument('--kd_epochs', type=int, default= 3, help="SEED")
parser.add_argument('--patch_div_loss',action='store_true', help = 'whether using score as a final logit')
parser.add_argument('--ewc_lambda', type=float, default=0.0, help='EWC regularization strength (0=off)')
parser.add_argument('--use_fisher', action='store_true', help='Use Fisher-weighted EWC (default: simple L2 anchor)')
parser.add_argument('--reg_lambda_eucl', type=float, default=0.1, help='L2 anchor strength for Euclidean params (features, add_on_layers, last_layer); 0=off')
parser.add_argument('--reg_lambda_proj', type=float, default=0.01, help='Projection-metric anchor strength for prototype_vectors (Grassmann); 0=off')
parser.add_argument('--min_per_label', type=int, default= 0, help="minimum number of image per each label")


args = parser.parse_args()
num_classes = args.num_classes
num_channels = args.num_channels
prototype_per_class = 3
prototype_shape = (num_classes, num_channels, 1, 1)  # * prototype_per_class
#prototype_shape = (num_classes*prototype_per_class, num_channels)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

train_batch_size = 80
test_batch_size = 64
train_push_batch_size =75

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-4,
                      'prototype_vectors': 2e-4}

last_layer_optimizer_lr = 1e-3


coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': 0.08,
    'l1': 1e-3,           
    'spatial_ent': 0.1,
    'sub_sep': -1e-7,
}
