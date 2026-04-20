import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.linalg as la
from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from models.wresnet import *
from util.receptive_field import compute_proto_layer_rf_info_v2
from utils.utils import *
from pymanopt.manifolds import Grassmann

base_architecture_to_features = {#'wresnet28x2': wresnet28x2,
                                 #'wresnet28x2': wide_resnet28x2_features,
                                 'resnet18': resnet18_features,
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



def init_Grassmann2(prototype_shape, num_classes, prototype_per_class):
    points = []
    basis= torch.nn.init.orthogonal_(torch.empty(prototype_shape[1],prototype_shape[1]))
    for i in range(num_classes):
        shuffled_indices = torch.randperm(prototype_shape[1])
        selected_numbers = shuffled_indices[:prototype_per_class]
        pre_eigen = torch.zeros((prototype_shape[1], prototype_shape[1]))
        for j in range(prototype_per_class):
            pre_eigen[selected_numbers[j], selected_numbers[j]]=1
        each_class_Grass=basis@pre_eigen@basis.T
        points.append(each_class_Grass)
    points = torch.stack(points) 
    #result = verify_projection_matrices(points, num_classes)
    #print('Verify projection matrices', result['trace'])
    return points.reshape(num_classes, prototype_shape[1],prototype_shape[1])  #conv2d로 내가 구현하지 않았기 때문에 굳이 4 dimension을 맞출 필요 X

def init_Grassmann(prototype_shape, num_classes, prototype_per_class):
    """
    Completely safe version that ensures leaf tensors
    """    
    # Create result tensor with the right shape first
    result = torch.zeros(num_classes,prototype_shape[1],prototype_shape[1])

    # Fill it with proper Grassmann points
    for i in range(num_classes):
        # Create temporary matrices for computation
        temp = torch.randn(prototype_shape[1], prototype_per_class)
        q, r = torch.linalg.qr(temp)
        basis = q[:, :prototype_per_class]
        
        # Compute projection matrix
        projection = torch.mm(basis, basis.t())  # Use torch.mm instead of @
        
        # Copy data directly into result tensor (no operations on result)
        result[i].copy_(projection)
    
    print(f"Created Grassmann tensor: shape={result.shape}, requires_grad={result.requires_grad}, is_leaf={result.is_leaf}")
    return result


class TESNet_Grassmann(nn.Module):
    def __init__(self, args, features, img_size, prototype_shape, dataset, prototype_per_class, proto_layer_rf_info,
                  num_classes, init_weights=True, 
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):    #proto_layer_rf_info

        super(TESNet_Grassmann, self).__init__()
        self.args = args
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.prototype_per_class = prototype_per_class
        self.num_prototypes = num_classes * prototype_per_class
        self.epsilon = 1e-4
        self.prototype_activation_function = prototype_activation_function #log
        self.dataset = dataset
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_classes,
                                                    self.num_classes)

        for j in range(self.num_classes):
            self.prototype_class_identity[j, j] = 1

        self.proto_layer_rf_info = proto_layer_rf_info
        #self.features = []
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
        if features_name.startswith('VGG') or features_name.startswith('RES') or features_name.startswith('WIDERES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base_architecture NOT implemented')
        # first_add_on_layer
        # conv2d-relu-conv2d-relu-sigmoid conv2d는 모두 1*1 
        if add_on_layers_type == 'bottleneck':      #여기는 딱히 압축의 영역이 아님
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels   #더 압축의 과정
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
        
        #self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              #requires_grad=True)
        self.prototype_vectors = nn.Parameter(init_Grassmann(self.prototype_shape, self.num_classes, self.prototype_per_class), requires_grad= True)
        print(f"Just created prototype_vectors: shape={self.prototype_vectors.shape}, is_leaf={self.prototype_vectors.is_leaf}")
        #self.prototype.requires_grad_(True)
        self.last_layer = nn.Linear(self.num_classes, self.num_classes, bias=False)
        print('TesNet construction finish')
        if init_weights:
            self._initialize_weights()

    
    def conv_features(self, x):

        temp = self.features(x)
        conv_features = self.add_on_layers(temp)

        return conv_features

    def _cosine_convolution(self, x): 

        x = F.normalize(x,p=2,dim=1)
        N,Ch,W,H=x.size()
        prototype_reshaped = self.prototype_vectors.reshape(self.num_classes, Ch, Ch)
        scores = torch.einsum('ncwh,lic->nliwh', x, prototype_reshaped)
        #distances = torch.einsum('ncwh,lck->nlkwh', x, prototype_reshaped)        
        scores = torch.squeeze(torch.norm(scores, p=2, dim=2)) #.view(N, self.num_classes, Ch, W * H)
        scores = scores.reshape(N, self.num_classes, -1) 
        distances = -scores  #input: (B.S, Ch, H, W)  prototype_vectors: after(C, Ch, Ch, 1), 여기 distance는 작을수록 좋음 3/17

        return distances
        
    def _project2basis(self, x):

        #x = F.normalize(x,p=2,dim=1)
        #norm_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=2)
        N,Ch,W,H = x.size()
        prototype_reshaped = self.prototype_vectors.reshape(self.num_classes, Ch, Ch)
        projected_vectors = torch.einsum('ncwh,lic->nliwh', x, prototype_reshaped)   #.view(N, self.num_classes, Ch, W * H)
        distances = torch.squeeze(torch.norm(projected_vectors, p=2, dim=2))
        distances = distances.reshape(N, self.num_classes, -1)  #얘는 클수록 좋음'
        return distances

    def prototype_distances(self, x):

        conv_features = self.conv_features(x)  #(N,C,W,H)
        project_distances = self._project2basis(conv_features)   # + distance, max pooling의 입력, projected_vectors size (N, C, 3)
        cosine_distances = self._cosine_convolution(conv_features)    #size(B,C,H,W)   25.11.18

        return project_distances, cosine_distances  #conv_features

    def global_min_pooling(self, distances):   #기존: (N,M*C,H,W) 내 코드: (N,C,H,W)

        #min_distances = -F.max_pool2d(-distances,
                                      #kernel_size=(distances.size()[2],
                                                   #distances.size()[3]))
        min_distances, _ = torch.topk(-distances, self.args.patch_num, dim=2)  #shape: (N,C,3)      
        min_distances = torch.squeeze(torch.mean(min_distances, dim=2)).reshape(-1, self.num_classes)    #shape:(N,C)                                     

        return min_distances

    def global_max_pooling(self, distances):    # projected_vectors

        #max_distances = F.max_pool2d(distances,
                                      #kernel_size=(distances.size()[2],
                                                   #distances.size()[3]))  #patch 중 최댓값을 고르는 방식, cnn feature map 크기와 똑같은 kernel size의 kernel 사용
        #max_distances = max_distances.reshape(-1, self.num_prototypes)
        #print('distances',distances)
        #if isinstance(distances, tuple):
            #distances = distances[0]  # Extract the tensor from the tuple
        max_distances, _ = torch.topk(distances, self.args.patch_num, dim=2)
        summed_distances = torch.squeeze(torch.mean(max_distances, dim=2)).reshape(-1, self.num_classes)
        #(1,2000)의 size인 max_distances?
        return summed_distances      # vectored_sel_projected_vectors

    def push_forward(self, x):

        conv_output = self.conv_features(x) #[batchsize,128,14,14]
        scores = self._project2basis(conv_output)
        distances = -scores    #여기에 왜 -를 하지????  25.03.21
        N,Ch,W,H = conv_output.size()
        distances = distances.reshape(N, self.num_classes, W, -1)
        return conv_output, distances

    def forward(self, x):

        project_distances, cosine_min_distances = self.prototype_distances(x)    #, cosine_min_distances
        project_max_distances = self.global_max_pooling(project_distances)  #(N,C)
        cosine_min_distances = self.global_min_pooling(cosine_min_distances)
        #project_min_distances = self.global_min_pooling(-project_distances)
        #project_max_distances, sel_projected_vectors = self.global_max_pooling(project_distances)  #(N,C)
        logits = self.last_layer(project_max_distances)   
        return  logits, cosine_min_distances, project_max_distances, project_distances    #, input_dim    #, sel_projected_vectors

    def get_score_vecs(self, input):
        #max_dist = (self.prototype.size[1]
                                    #* self.prototype.size[2]
                                    #* self.prototype.size[3])
        temp, feature_map= self.conv_features(input)
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


class GrassmannManifoldOptimizer(optim.Optimizer):
    def __init__(self, model, optimizer_specs, prototype_per_class):
        self.model = model
        
        # Initialize base optimizer
        super(GrassmannManifoldOptimizer, self).__init__(self.model.module.parameters(),defaults = {})
        self.prototype_per_class = prototype_per_class
        # Separate Adam optimizers for non-prototype parameters
        self.adam_optimizers = []
        self.proto_group = None

        for group in optimizer_specs:
            if group.get('name') == 'prototypes':
                self.proto_group = group # Save the prototype group
            else:
                self.adam_optimizers.append(optim.Adam(
                group['params'],
                #params_list,
                lr=group.get('lr', 0.001), # Add default
                weight_decay=group.get('weight_decay', 0) # Add default
            ))

                    
        self.manifold = Grassmann(self.model.module.prototype_shape[1], self.prototype_per_class)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.proto_group:
            self.update_prototype(self.proto_group['params'][0], self.proto_group['lr'])

        # Step all other optimizers
        for adam_opt in self.adam_optimizers:
            adam_opt.step()
    
    def update_prototype(self, param: torch.Tensor, lr: float):
        """
        Projection-matrix update (batched, loop-free over classes) using
        a project-then-retract method.
        param: [C, n, n] stores projection matrices P = Y Y^T (rank k).
        lr: step size.
        """
        assert param.grad is not None, "Call loss.backward() before update_prototype."
        assert param.ndim == 3 and param.shape[-1] == param.shape[-2], \
            f"Expected [C, n, n], got {tuple(param.shape)}"

        k = int(self.prototype_per_class)  # target rank
        C, n, _ = param.shape

        with torch.no_grad():
            # Current projection and its Euclidean gradient (symmetrize for stability)
            P = param    # 0.5 * + param.transpose(-1, -2))                 # [C, n, n]
            G = param.grad     # 0.5 * (+ param.grad.transpose(-1, -2))       # [C, n, n]

            # --- START: MODIFICATION ---
            # We replace the simple Euclidean step with a projection based on Eq. 3.3
            
            # Create Identity matrix for the batch
            I = torch.eye(n, device=param.device).unsqueeze(0).expand(C, n, n) # [C, n, n]
            
            # Calculate (I - P)
            I_minus_P = I - P

            # Calculate the two terms of Eq. 3.3: (I-P)SP + PS(I-P)
            # [cite: 1456-1457]
            # Here S = G (the Euclidean gradient)
            term1 = torch.bmm(torch.bmm(I_minus_P, G), P) # [C, n, n]
            term2 = torch.bmm(torch.bmm(P, G), I_minus_P) # [C, n, n]

            # G_riemann is the projected Riemannian gradient
            G_riemann = term1 + term2
            
            # Now we take a step in the direction of the Riemannian gradient
            P_tilde = P - lr * G_riemann
            # --- END: MODIFICATION ---

            # Spectral projection back to the manifold of rank-k projectors:
            # This part is the retraction and remains the same as your file [cite: 798-800]
            evals, evecs = torch.linalg.eigh(P_tilde)                   # evals: [C, n], evecs: [C, n, n]
            topk_idx = torch.topk(evals, k, dim=-1).indices             # [C, k]
            
            # Gather the corresponding eigenvector columns
            gather_idx = topk_idx.unsqueeze(-2).expand(C, n, k)         # [C, n, k]
            Y = torch.gather(evecs, dim=-1, index=gather_idx)           # [C, n, k]

            # Rebuild the projection matrices: P_new = Y Y^T (exactly symmetric/idempotent, rank k)
            P_new = Y @ Y.transpose(-1, -2)                             # [C, n, n]

            # In-place writeback
            param.copy_(P_new)

def construct_TesNet(args, base_architecture, prototype_per_class, dataset = 'cifar10', pretrained=True, img_size=224, 
                    prototype_shape=(200, 128, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,#224
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return TESNet_Grassmann(args = args, features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape, dataset = dataset, 
                 prototype_per_class= prototype_per_class, 
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

