import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util.helpers import list_of_distances
from St_model import *
from settings_CUB_Centralized import last_layer_optimizer_lr   #joint_optimizer_lrs, 
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

def _train_or_test(args, client_idx, client_model, X_data, y_data, body_train, is_train, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    client_dataset = MyDataset(X_data, y_data)
    client_model = client_model.train()
    client_model = client_model.cuda()

    if body_train:
        if args.dataset == 'CUB' or 'Stanford_dog':
            #initial_joint_optimizer_lrs =  {'features': 5e-5,'add_on_layers':  1e-3,'prototype_vectors': 1e-3}
            initial_joint_optimizer_lrs =  {'features': 1e-4,'add_on_layers':  3e-3,'prototype_vectors': 3e-3}

        else:
            initial_joint_optimizer_lrs = {'features': 1e-5,'add_on_layers':  2e-4,'prototype_vectors': 2e-4}
        
        joint_optimizer_lrs = {}
        for i,j in initial_joint_optimizer_lrs.items():
            j *= 0.1**(args.epoch//40)
            joint_optimizer_lrs[i] = j
        joint_optimizer_lrs = initial_joint_optimizer_lrs

        optimizer_specs = beside_last(joint_optimizer_lrs, client_model)
        train_epochs = args.SL_epochs
        dataloader = DataLoader(
        client_dataset,
        batch_size=args.local_bs,
        num_workers=4,
        pin_memory=False)
    else:
        optimizer_specs = last_only(last_layer_optimizer_lr, client_model)
        train_epochs = args.fine_tune_epochs
        dataloader = DataLoader(
        client_dataset,
        batch_size= 256,
        num_workers=4,
        pin_memory=False)

    optimizer = StiefelManifoldOptimizer(client_model, optimizer_specs, client_model.module.prototype_per_class) 
    dataloader = DataLoader(
        client_dataset,
        batch_size=args.local_bs,
        num_workers=4,
        pin_memory=False)
    cluster_cost=torch.tensor(0)
    separation_cost=torch.tensor(0)

    for epoch in range(train_epochs):
        total_cross_entropy = 0
        total_cluster_cost = 0
        total_orth_cost = 0
        total_subspace_sep_cost = 0
        total_separation_cost = 0
        total_avg_separation_cost = 0
        total_loss=0
        total_l1 = 0 
        n_examples = 0
        n_correct = 0
        n_batches = 0
        for i, (image, label) in enumerate(dataloader):
            image = image.cuda()
            target = label.cuda()
            client_model = client_model.cuda()
            client_model.module.prototype_class_identity= client_model.module.prototype_class_identity.cuda()
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                output, min_distances = client_model(image)
                del image
                cross_entropy = torch.nn.functional.cross_entropy(output, target)
                cur_basis_matrix = torch.squeeze(client_model.module.prototype_vectors) #[2000,128]
                subspace_basis_matrix = cur_basis_matrix.reshape(client_model.module.num_classes, client_model.module.num_prototypes_per_class, client_model.module.prototype_shape[1])#[200,10,128]
                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2)
                projection_operator_1 = torch.unsqueeze(subspace_basis_matrix, dim=0)
                projection_operator_2 = torch.unsqueeze(subspace_basis_matrix_T, dim = 1)
                del subspace_basis_matrix_T

                projection_operator = torch.matmul(projection_operator_1,projection_operator_2)#[200,128,10] [200,10,128] -> [200,128,128]
                pairwise_distance= torch.norm(projection_operator, dim = [2,3])
                subspace_sep = torch.norm(client_model.module.prototype_per_class-pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()

                del projection_operator_1
                del projection_operator_2
                del pairwise_distance
                if class_specific:
                    prototypes_of_correct_class = torch.t(client_model.module.prototype_class_identity[:,target]).cuda()  #(N,M*C)
                    #inverted_distances, _ = torch.min((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                    #cluster_cost = torch.mean(max_dist - inverted_distances)
                    inverted_distances, _ = torch.max((min_distances)*prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(- inverted_distances)

                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(- inverted_distances_to_nontarget_prototypes)
                    
                if use_l1_mask:
                    l1_mask = 1 - torch.t(client_model.module.prototype_class_identity).cuda()
                    l1 = (client_model.module.last_layer.weight * l1_mask).norm(p=1)
                    # weight 200,2000   prototype_class_identity [2000,200]
                else:
                    l1 = client_model.module.last_layer.weight.norm(p=1)

            if is_train:
                if not class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy + coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep)   
                        #coefs['crs_ent'] * cross_entropy+ coefs['clst'] * cluster_cost+ coefs['sep'] * separation_cost
                                        #+ coefs['l1'] * l1 + coefs['orth'] * orth_cost+ coefs['sub_sep'] * subspace_sep
                    else:
                        #loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1 * orth_cost - 1e-7 * subspace_sep
                        loss = cross_entropy + 1e-4 * l1 -1e-7 * subspace_sep
                else:
                    if coefs is not None: 
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1
                            + coefs['sub_sep'] * subspace_sep)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost - 1e-7 * subspace_sep

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            del min_distances

            if epoch == train_epochs-1:
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()
                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_subspace_sep_cost += subspace_sep.item()
                total_separation_cost += separation_cost.item()
                total_l1 += l1.item()
                total_loss+= loss.item()
                del target
                del output
                del predicted

    log('\tclient_idx: \t{0}'.format(client_idx))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\torth: \t{0}'.format(total_orth_cost / n_batches))
    log('\tsubspace_sep: \t{0}'.format(total_subspace_sep_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(client_model.module.last_layer.weight.norm(p=1).item()))
    log('\tl1_loss: \t\t{0}'.format(total_l1/n_batches))
    log('\taverage loss: \t\t{0}'.format(total_loss/ n_batches))
 
    if body_train:
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,
                        'cluster_loss': total_cluster_cost / n_batches,                            
                        'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                        'separation_loss': total_separation_cost / n_batches,
                        'avg_separation': total_avg_separation_cost / n_batches,
                        'l1':client_model.module.last_layer.weight.norm(p=1).item(),
                        'l1_loss' : (total_l1/n_batches),
                        'accu' : n_correct/n_examples,
                        'average loss': (total_loss/ n_batches)}

    else:
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,
                        'cluster_loss': total_cluster_cost / n_batches,                            
                        'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                        'separation_loss': total_separation_cost / n_batches,
                        'avg_separation': total_avg_separation_cost / n_batches,
                        'l1': client_model.module.last_layer.weight.norm(p=1).item(),
                        'l1_loss' : (total_l1/n_batches),
                        'accu' : n_correct/n_examples,
                        'average loss': (total_loss/ n_batches)}
            
    return client_model, result_loss

def local_test_global_model(args, client_model, X_test, y_test, coefs):   #이건 without classifier 버전.
    
    total_cross_entropy = 0
    total_subspace_sep_cost = 0
    total_separation_cost = 0
    total_loss = 0
    total_l1 = 0 
    n_examples = 0
    n_correct = 0
    n_batches = 0
    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.local_bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda()
            target = label.cuda()                  
            client_model = client_model.cuda()
            client_model.prototype_class_identity= client_model.prototype_class_identity.cuda()
            output, min_distances = client_model(image)
            del image
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            prototypes_of_correct_class = torch.t(client_model.prototype_class_identity[:,target]).cuda()  #(N,M*C)
            #inverted_distances, _ = torch.min((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            #cluster_cost = torch.mean(max_dist - inverted_distances)
            inverted_distances, _ = torch.max((min_distances)*prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(- inverted_distances)

            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(- inverted_distances_to_nontarget_prototypes)

            cur_basis_matrix = torch.squeeze(client_model.prototype_vectors) #[2000,128]
            subspace_basis_matrix = cur_basis_matrix.reshape(client_model.num_classes, client_model.num_prototypes_per_class, client_model.prototype_shape[1])#[200,10,128]
            subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2)
            projection_operator_1 = torch.unsqueeze(subspace_basis_matrix, dim=0)
            projection_operator_2 = torch.unsqueeze(subspace_basis_matrix_T, dim = 1)
            del subspace_basis_matrix_T

            projection_operator = torch.matmul(projection_operator_1,projection_operator_2)#[200,128,10] [200,10,128] -> [200,128,128]
            pairwise_distance= torch.norm(projection_operator, dim = [2,3])
            subspace_sep = torch.norm(client_model.prototype_per_class-pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()

            del projection_operator_1
            del projection_operator_2
            del pairwise_distance

            l1_mask = 1 - torch.t(client_model.prototype_class_identity).cuda()
            l1 = (client_model.last_layer.weight * l1_mask).norm(p=1)

            loss = coefs['crs_ent'] * cross_entropy + coefs['clst'] * cluster_cost + coefs['sep'] * separation_cost +coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep                        
            del min_distances
            
            #softmax_scores = F.softmax(project_max_distances, dim=1)
            #print('Project max distances', project_max_distances[0,1:20])
            #print('Test Output logit', output[0,1:20])
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_subspace_sep_cost += subspace_sep.item()
            total_separation_cost += separation_cost.item()
            total_l1 += l1.item()
            total_loss+= loss.item()
            del target
            del output
            del predicted
            torch.cuda.empty_cache()
            
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,                       
                    'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'l1':client_model.last_layer.weight.norm(p=1).item(),
                    'l1_loss' : (total_l1/n_batches),
                    'accu' : n_correct/n_examples,
                    'average loss': (total_loss/ n_batches)}

    return result_loss

def fine_tune_train(args, client_idx, client_model, X_data, y_data, is_train= True, body_train=False, coefs=None, log=print):
    
    log('\tfine_tune_train')
    client_after, loss_dict = _train_or_test(args, client_idx, client_model, X_data, y_data, body_train, is_train, class_specific=True, use_l1_mask=True,
                   coefs=coefs, log=log)

    return client_after, loss_dict

def train(args, client_idx, client_model, X_data, y_data, is_train = True, body_train = True, coefs=None, log=print):

    log('\ttrain')
    client_after, loss_dict = _train_or_test(args, client_idx, client_model, X_data, y_data, body_train, is_train, class_specific=True, use_l1_mask=True,
                   coefs=coefs, log=log)
    #results = verify_projection_matrices(client_model.module.prototype_vectors, client_model.module.num_classes)
# Corrected code with one argument
    #log(f'client_idx{client_idx}: Grassmann_verification {results["summary"]}')
    return client_after, loss_dict


def last_only(last_layer_optimizer_lr, model, log=print):
    #for p in model.module.features.parameters():
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    optimizer_specs =  [{'params': model.module.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    log('\tlast layer')

    return optimizer_specs

def warm_only(warm_optimizer_lrs, model, log=print):
    for p in model.features.parameters(): 
        p.requires_grad = False
    for p in model.add_on_layers.parameters():  #순수 여기만 하는건 뭐지?
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True  
    for p in model.last_layer.parameters():
        p.requires_grad = False
    optimizer_specs = [{'params': model.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']}]
    log('\tlast layer')
    log('\twarm')

    return optimizer_specs

def beside_last(joint_optimizer_lrs, model):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = False

    # Convert generators to lists to prevent single-use consumption
    features_params = list(model.module.features.parameters())
    add_on_layers_params = list(model.module.add_on_layers.parameters())

    optimizer_specs = [
        {'params': features_params, 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3, 'name': 'features'},
        {'params': add_on_layers_params, 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3, 'name': 'add_on'},
        {'params': [model.module.prototype_vectors], 'lr': joint_optimizer_lrs['prototype_vectors'], 'name': 'prototypes'}
    ]

    return optimizer_specs


def construct_TesNet(base_architecture, dataset = 'cifar10', pretrained=True, img_size=224,
                    prototype_shape=(360, 64), num_classes=120,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,#224
                                                         layer_filter_sizes=layer_filter_sizes,#
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=1)
    return TESNet_Stiefel(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape, dataset = dataset, 
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

