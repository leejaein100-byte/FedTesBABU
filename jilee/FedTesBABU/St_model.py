import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import qr
from utils.utils import *
from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from models.wresnet import *

from util.receptive_field import compute_proto_layer_rf_info_v2
from pymanopt.manifolds import Stiefel

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}
def generate_stiefel(n, k):
    """
    Generate a point on the Stiefel manifold St(k,n)
    
    Parameters:
    n (int): Dimension of the ambient space
    k (int): Number of orthonormal vectors (k <= n)
    
    Returns:
    numpy.ndarray: A point on the Stiefel manifold of shape (n, k)
    """
    if k > n:
        raise ValueError("k must be less than or equal to n")
    
    # Generate random matrix
    A = torch.randn(n, k)
    
    # QR decomposition
    Q, R = torch.linalg.qr(A)
    
    # Take first k columns of Q to get point on Stiefel manifold
    return Q[:, :k]

def initialize_multiple_stiefel(prototype_shape, num_classes):
    """
    Initialize multiple points on the Stiefel manifold
    
    Parameters:
    n (int): Dimension of the ambient space
    k (int): Number of orthonormal vectors (k <= n)
    num_points (int): Number of points to generate
    
    Returns:
    list: List of numpy arrays, each representing a point on the Stiefel manifold
    """
    #print('1')
    n = prototype_shape[1]
    num_points = num_classes
    k = prototype_shape[0]//num_classes
    manifolds = []
    for _ in range(num_points):
        point = generate_stiefel(n, k)
        #print('2')
        manifolds.append(point)
    manifolds = torch.stack(manifolds)
    #print('3')
    manifolds = manifolds.reshape(k*num_points, prototype_shape[1])
    while len(manifolds.size()) != len(prototype_shape):
        manifolds= manifolds.unsqueeze(-1)
    
    return manifolds

class TESNet_Stiefel(nn.Module):
    def __init__(self, features, img_size, prototype_shape, dataset, 
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(TESNet_Stiefel, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.prototype_per_class = self.num_prototypes // self.num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function #log
        self.dataset = dataset
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info
        if self.dataset == 'cifar10' or self.dataset == 'cifar100' or self.dataset =='tiny_imagenet' :
            self.features = []
            for name, module in features.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if self.dataset == 'cifar10' or self.dataset == 'cifar100':
                    print('dataset name', self.dataset)
                    if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d) and not name == 'conv5':
                        self.features.append(module)
                elif self.dataset == 'tiny_imagenet' or self.dataset == 'stl10':
                    if not isinstance(module, nn.Linear):
                        self.features.append(module)
            #print(self.features)            
            self.features = nn.Sequential(*self.features)
        else:
            self.features = features
        
        features_name = str(features).upper()
        print('features_name', features_name)
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        #print('4')
        self.prototype_vectors = nn.Parameter(initialize_multiple_stiefel(self.prototype_shape, self.num_classes), requires_grad = True)
        print(f"Just created prototype_vectors: shape={self.prototype_vectors.shape}, is_leaf={self.prototype_vectors.is_leaf}")
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        print('TesNet construction finish')
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):

        x = self.features(x)
        x = self.add_on_layers(x)

        return x

    def _cosine_convolution(self, x):

        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -distances

        return distances
    def _project2basis(self,x):

        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        return distances

    def prototype_distances(self, x):

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)

        return project_distances, cosine_distances

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def global_min_pooling(self,distances):

        min_distances = F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)

        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances

    def forward(self, x):

        project_distances, cosine_distances = self.prototype_distances(x)
        cosine_min_distances = self.global_min_pooling(cosine_distances)
        project_max_distances = self.global_max_pooling(project_distances)
        prototype_activations = project_max_distances
        logits = self.last_layer(prototype_activations)
        return logits, cosine_min_distances

    def push_forward(self, x):

        conv_output = self.conv_features(x) #[batchsize,128,14,14]

        distances = self._project2basis(conv_output)
        distances = - distances
        return conv_output, distances

    def get_score_vecs(self, input):
        #max_dist = (self.prototype.size[1]
                                    #* self.prototype.size[2]
                                    #* self.prototype.size[3])
        feature_map= self.conv_features(input)
        distances = self._project2basis(feature_map)   #여기서 distances는 -가 곱해진 것, 각 feature map patch와 prototype의 모든 경우의 수의 distance
        max_score = self.global_max_pooling(distances)  # -의 min이라서 max를 뽑는 느낌
        return max_score

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


class StiefelManifoldOptimizer(optim.Optimizer):
    def __init__(self, model, optimizer_specs, prototype_per_class):
        self.model = model
        
        # [cite_start]Handle DataParallel [cite: 145]
        if hasattr(model, 'module'):
            self.actual_model = model.module
        else:
            self.actual_model = model [cite: 146]
            
        # Initialize base optimizer
        super(StiefelManifoldOptimizer, self).__init__(self.actual_model.parameters(), defaults={}) 
        
        self.prototype_per_class = prototype_per_class
        
        # Separate Adam optimizers for non-prototype parameters
        self.adam_optimizers = []
        self.proto_group = None

        # [cite_start]Use name-based parameter group identification [cite: 147]
        for group in optimizer_specs:
            if group.get('name') == 'prototypes':
                self.proto_group = group # Save the prototype group [cite: 147]
            else:
                self.adam_optimizers.append(optim.Adam(
                    group['params'],
                    lr=group.get('lr', 0.001), # Add default [cite: 148]
                    weight_decay=group.get('weight_decay', 0) # Add default [cite: 148]
                ))

        if self.proto_group is None:
            print("WARNING: Prototype group not found in optimizer_specs!")
        
        self.manifold = Stiefel(
            n=self.actual_model.prototype_shape[1],
            p=self.prototype_per_class,
            k=self.actual_model.num_classes
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): 
                loss = closure()

        if self.proto_group:
            prototype_tensor = self.proto_group['params'][0] 
            self.update_prototype(prototype_tensor, self.proto_group['lr']) 
        
        # [cite_start]Step all other optimizers [cite: 151]
        for adam_opt in self.adam_optimizers:
            adam_opt.step() 
            
        return loss

    # ---
    # **MODIFICATION 2: Vectorized update_prototype (no loop)**
    # ---
    @torch.no_grad()
    def update_prototype(self, param, lr):
        """
        Performs a batched Stiefel manifold update for all classes at once.
        'param' is the full prototype tensor with shape [C, n, p].
        """
        # Check for gradient on the whole tensor
        if param.grad is None:
            print("WARNING: Prototype tensor has no gradient. Skipping update.")
            return    

        k = self.actual_model.num_classes
        n = self.actual_model.prototype_shape[1]
        p = self.prototype_per_class
        
        # Squeeze the 4D tensor (k*p, n, 1, 1) to (k*p, n)
        param_squeezed = param.data.squeeze()
        grad_squeezed = param.grad.data.squeeze()

        # Reshape (k*p, n) to (k, p, n) and then permute to (k, n, p)
        try:
            param_3d = param_squeezed.reshape(k, p, n).permute(0, 2, 1)
            grad_3d = grad_squeezed.reshape(k, p, n).permute(0, 2, 1)
        except RuntimeError as e:
            print(f"Error during reshape. Tensor shape: {param_squeezed.shape}, k={k}, p={p}, n={n}")
            raise e

        # Convert the 3D tensors to NumPy for pymanopt
        cur_param_np = param_3d.detach().cpu().numpy()
        cur_grad_np = grad_3d.detach().cpu().numpy()

        # Perform the batched (k=C) manifold operations
        param_grad = self.manifold.projection(cur_param_np, cur_grad_np)
        new_param = self.manifold.retraction(cur_param_np, -lr * param_grad)

        # --- START FIX ---

        # Convert the 3D numpy result back to a tensor (k, n, p)
        # Also, fix the bug from the source file: new_param_tensor -> torch.from_numpy(new_param)
        new_param_tensor_3d = torch.from_numpy(new_param).to(param.device) 
        
        # Permute (k, n, p) back to (k, p, n)
        new_param_permuted = new_param_tensor_3d.permute(0, 2, 1)
        
        # Reshape (k, p, n) to (k*p, n)
        new_param_squeezed = new_param_permuted.reshape(k * p, n)

        # Unsqueeze to the original 4D shape (k*p, n, 1, 1) and copy
        param.data.copy_(new_param_squeezed.unsqueeze(-1).unsqueeze(-1))
        
        # --- END FIX ---