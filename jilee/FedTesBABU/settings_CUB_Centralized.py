import argparse

parser = argparse.ArgumentParser(description='Centralized TesNet Training')

parser.add_argument('--img_size',type=int, default=224)
parser.add_argument('--Device_id', type=str, default='0', help='GPU ID to use')
parser.add_argument('--arch', type=str, default='resnet50', help='Model architecture')
parser.add_argument('--num_channels', type=int, default=64, help='Number of channels')
parser.add_argument('--dataset', type=str, default="Stanford_dog", help='Dataset name')
parser.add_argument('--num_train_epochs', type=int, default=80, help="Number of training epochs")
parser.add_argument('--prototype_per_class', type=int, default=3, help="Number of prototypes per class")
parser.add_argument('--num_classes', type=int, default=120, help="Number of classes")
parser.add_argument('--tr_frac', type=float, default=0.8, help='model name')
parser.add_argument('--seed', type=int, default= 42, help="SEED")
parser.add_argument('--fine_tune_interval', type=int, default=10, help="fine tune update interval")
parser.add_argument('--push_interval', type=int, default=1, help="push update interval")
parser.add_argument('--fine_tune_epochs', type=int, default=10, help="fine tune epochs")
parser.add_argument('--use_bbox', type=bool, default=True,
                    help='Use bounding box cropping')
parser.add_argument('--local_bs', type=int, default=128, help="local batch size")
parser.add_argument('--update_interval', type=int, default=10)
parser.add_argument('--SL_epochs',type=int, default=3)
parser.add_argument('--alpha',type=float, default=0.5)
parser.add_argument('--iid', action='store_true', help="Use IID distribution")
parser.add_argument('--num_users', type=int, default=8, help="number of users")
parser.add_argument('--server_id_size', type=int, default = 3328, help="server_id size")
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--min_per_label', type=int, default = 0)


args = parser.parse_args()
img_size = 224
num_classes = args.num_classes
num_channels = args.num_channels
prototype_per_class = args.prototype_per_class
prototype_shape = (num_classes * prototype_per_class, num_channels, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'
train_batch_size = 128
test_batch_size = 100
train_push_batch_size =75

joint_optimizer_lrs = {'features':  1e-4,    # 괜찮았던 FL lr: 3e-6,    원 논문: #1e-4
                       'add_on_layers':  2e-3,   #8e-5,      #3e-3
                       'prototype_vectors': 2e-3}     #8e-5    #3e-3
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-4,
                      'prototype_vectors': 2e-4}

#last_layer_optimizer_lr = 2e-6    # 1e-4
last_layer_optimizer_lr = 1e-3

#coefs = {
#    'crs_ent': 1,
#    'clst': 0.8,
 #   'sep': -0.08,
 #   'l1': 1e-4,
 #   'orth': 1e-4,
 #   'sub_sep': -1e-7,
#}
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'orth': 1e-4,
    'sub_sep': -1e-7,
}
num_train_epochs = args.num_train_epochs
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % args.fine_tune_interval == 0]


