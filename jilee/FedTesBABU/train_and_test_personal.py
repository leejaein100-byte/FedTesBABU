import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.misc import *
from utils.utils import *

def _train_or_test(args, client_idx, client_model, X_data, y_data, body_train, is_train, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):

    client_dataset = MyDataset(X_data, y_data)
    client_model = client_model.train()
    client_model = client_model.cuda()

    # --- Optimizer Setup (Adam instead of Grassmann) ---
    if body_train:
        # Use learning rates from args if available, else defaults
        lrs = args.joint_optimizer_lrs if hasattr(args, 'joint_optimizer_lrs') else {'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
        optimizer_specs = beside_last(lrs, client_model)
        train_epochs = args.SL_epochs
    else:
        lr = args.last_layer_optimizer_lr if hasattr(args, 'last_layer_optimizer_lr') else 1e-4
        optimizer_specs = last_only(lr, client_model) 
        train_epochs = args.fine_tune_epochs

    # Replace Grassmann optimizer with Adam
    optimizer = torch.optim.Adam(optimizer_specs)

    dataloader = DataLoader(
        client_dataset,
        batch_size=args.local_bs,
        num_workers=4,
        pin_memory=False)
    
    # Initialize trackers
    cluster_cost = torch.tensor(0.0).cuda()
    separation_cost = torch.tensor(0.0).cuda()
    subspace_sep = torch.tensor(0.0).cuda()
    orth_cost = torch.tensor(0.0).cuda()

    for epoch in range(train_epochs):
        total_cross_entropy = 0
        total_cluster_cost = 0
        total_subspace_sep_cost = 0
        total_separation_cost = 0
        total_orth_cost = 0
        total_loss = 0
        total_l1 = 0 
        n_examples = 0
        n_correct = 0
        n_batches = 0
        
        for i, (image, label) in enumerate(dataloader):
            image = image.cuda()
            target = label.cuda()  
            client_model = client_model.cuda()
            
            # Handle DataParallel wrapping for attribute access
            net = client_model.module if hasattr(client_model, 'module') else client_model
            net.prototype_class_identity = net.prototype_class_identity.cuda()
            
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                # Forward pass
                output, cosine_min_distances, project_max_distances = client_model(image)
                
                del image
                cross_entropy = torch.nn.functional.cross_entropy(output, target)
                
                # --- Geometry Calculations (Orthogonality & Separation) ---
                # Shape: [2000, 128, 1, 1] -> [2000, 128]
                protos = net.prototype_vectors.squeeze() 
                num_classes = net.num_classes
                proto_per_class = net.prototype_per_class
                dim = protos.shape[1]
                
                # Reshape to [Num_Classes, Num_Proto_Per_Class, Dim] -> [200, 10, 128]
                # Then Transpose to [Num_Classes, Dim, Num_Proto_Per_Class] -> [200, 128, 10]
                protos_reshaped = protos.view(num_classes, proto_per_class, dim)
                subspace_basis = protos_reshaped.permute(0, 2, 1) 
                subspace_basis_T = torch.transpose(subspace_basis, 1, 2) 
                
                # 1. Orthogonality Cost (orth_cost)
                # Goal: Basis vectors within a class should be orthogonal (U^T * U = I)
                # [200, 10, 128] * [200, 128, 10] -> [200, 10, 10]
                orth_operator = torch.matmul(subspace_basis_T, subspace_basis) 
                I_operator = torch.eye(proto_per_class, device=orth_operator.device).unsqueeze(0).expand(num_classes, -1, -1)
                difference_value = orth_operator - I_operator
                orth_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1,2]) - 0))

                # 2. Subspace Separation Cost (subspace_sep)
                # Goal: Subspaces should be far apart (Projection matrices P = U * U^T should differ)
                projection_operator = torch.matmul(subspace_basis, subspace_basis_T) # [200, 128, 128]
                projection_operator_1 = torch.unsqueeze(projection_operator, dim=1) # [200, 1, 128, 128]
                projection_operator_2 = torch.unsqueeze(projection_operator, dim=0) # [1, 200, 128, 128]
                
                pairwise_distance = torch.norm(projection_operator_1 - projection_operator_2 + 1e-10, p='fro', dim=[2,3])
                subspace_sep = torch.norm(pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2, dtype=torch.double)).cuda()

                del projection_operator, projection_operator_1, projection_operator_2, pairwise_distance, orth_operator, difference_value

                # --- Cluster & Separation Costs ---
                if class_specific:
                    prototypes_of_correct_class = torch.t(net.prototype_class_identity[:, target]).cuda()
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    
                    # Cluster Cost: Minimize distance to correct prototype
                    masked_cluster = (cosine_min_distances).masked_fill(prototypes_of_correct_class == 0, float('inf'))
                    min_dist_correct, _ = torch.min(masked_cluster, dim=1)
                    cluster_cost = torch.mean(min_dist_correct) 

                    # Separation Cost: Maximize distance to wrong prototype (Minimize negative)
                    masked_sep = (cosine_min_distances).masked_fill(prototypes_of_wrong_class == 0, float('-inf'))
                    min_dist_wrong, _ = torch.max(masked_sep, dim=1)
                    separation_cost = torch.mean(min_dist_wrong) 

                # --- L1 Regularization ---
                if use_l1_mask:
                    l1_mask = 1 - torch.t(net.prototype_class_identity).cuda()
                    l1 = (net.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = net.last_layer.weight.norm(p=1)

            # --- Optimization Step ---
            if is_train:
                if not class_specific:
                    loss = (coefs['crs_ent'] * cross_entropy + 
                            coefs['l1'] * l1 + 
                            coefs['sub_sep'] * subspace_sep + 
                            coefs['orth'] * orth_cost)
                else:
                    loss = (coefs['crs_ent'] * cross_entropy + 
                            coefs['clst'] * cluster_cost + 
                            coefs['sep'] * separation_cost + 
                            coefs['l1'] * l1 + 
                            coefs['sub_sep'] * subspace_sep + 
                            coefs['orth'] * orth_cost) # Include Orth cost
                   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Optional: Normalize prototypes after step to keep them on/near manifold
                # net.prototype_vectors.data = F.normalize(net.prototype_vectors, p=2, dim=1).data
            
            del cosine_min_distances, project_max_distances
            
            # --- Recording Stats (Last Epoch) ---
            if epoch == train_epochs - 1:
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()
                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_subspace_sep_cost += subspace_sep.item()
                total_separation_cost += separation_cost.item()
                total_orth_cost += orth_cost.item()
                total_l1 += l1.item()
                total_loss += loss.item()
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
    log('\tl1: \t\t{0}'.format(net.last_layer.weight.norm(p=1).item()))
    log('\taverage loss: \t\t{0}'.format(total_loss / n_batches))

    result_loss = {
        'cross_entropy': total_cross_entropy / n_batches,
        'cluster_loss': total_cluster_cost / n_batches,                            
        'subspace_sep_loss': total_subspace_sep_cost / n_batches,
        'separation_loss': total_separation_cost / n_batches,
        'orth_loss': total_orth_cost / n_batches,
        'l1': net.last_layer.weight.norm(p=1).item(),
        'accu': n_correct / n_examples,
        'average loss': (total_loss / n_batches)
    }
            
    return client_model, result_loss


# --- Wrapper Functions ---

def train(args, client_idx, client_model, X_data, y_data, is_train=True, body_train=True, coefs=None, log=print):
    log('\ttrain')
    client_after, loss_dict = _train_or_test(args, client_idx, client_model, X_data, y_data, 
                                             body_train, is_train, 
                                             class_specific=True, use_l1_mask=True,
                                             coefs=coefs, log=log)
    return client_after, loss_dict

def fine_tune_train(args, client_idx, client_model, X_data, y_data, is_train=True, body_train=False, coefs=None, log=print):
    log('\tfine_tune_train')
    client_after, loss_dict = _train_or_test(args, client_idx, client_model, X_data, y_data, 
                                             body_train, is_train, 
                                             class_specific=True, use_l1_mask=True,
                                             coefs=coefs, log=log)
    return client_after, loss_dict

def local_test_global_model(args, net, X_test, y_test, coefs):   #이건 without classifier 버전.
    
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
        num_workers=4,
        pin_memory=True)
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda()
            target = label.cuda()                  
            net = net.cuda()
            net.prototype_class_identity= net.prototype_class_identity.cuda()
            output, cosine_min_distances, project_max_distances = net(image)
            del image
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            protos = net.prototype_vectors.squeeze() 
            num_classes = net.num_classes
            proto_per_class = net.prototype_per_class
            dim = protos.shape[1]
            
            # Reshape to [Num_Classes, Num_Proto_Per_Class, Dim] -> [200, 10, 128]
            # Then Transpose to [Num_Classes, Dim, Num_Proto_Per_Class] -> [200, 128, 10]
            protos_reshaped = protos.view(num_classes, proto_per_class, dim)
            subspace_basis = protos_reshaped.permute(0, 2, 1) 
            subspace_basis_T = torch.transpose(subspace_basis, 1, 2) 
            
            # 1. Orthogonality Cost (orth_cost)
            # Goal: Basis vectors within a class should be orthogonal (U^T * U = I)
            # [200, 10, 128] * [200, 128, 10] -> [200, 10, 10]
            orth_operator = torch.matmul(subspace_basis_T, subspace_basis) 
            I_operator = torch.eye(proto_per_class, device=orth_operator.device).unsqueeze(0).expand(num_classes, -1, -1)
            difference_value = orth_operator - I_operator
            orth_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1,2]) - 0))

            # 2. Subspace Separation Cost (subspace_sep)
            # Goal: Subspaces should be far apart (Projection matrices P = U * U^T should differ)
            projection_operator = torch.matmul(subspace_basis, subspace_basis_T) # [200, 128, 128]
            projection_operator_1 = torch.unsqueeze(projection_operator, dim=1) # [200, 1, 128, 128]
            projection_operator_2 = torch.unsqueeze(projection_operator, dim=0) # [1, 200, 128, 128]
            
            pairwise_distance = torch.norm(projection_operator_1 - projection_operator_2 + 1e-10, p='fro', dim=[2,3])
            subspace_sep = torch.norm(pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2, dtype=torch.double)).cuda()

            del projection_operator, projection_operator_1, projection_operator_2, pairwise_distance, orth_operator, difference_value

            l1_mask = 1 - torch.t(net.prototype_class_identity).cuda()
            l1 = (net.last_layer.weight * l1_mask).norm(p=1)

            prototypes_of_correct_class = torch.t(net.prototype_class_identity[:, target]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            
            # Cluster Cost: Minimize distance to correct prototype
            masked_cluster = (cosine_min_distances).masked_fill(prototypes_of_correct_class == 0, float('inf'))
            min_dist_correct, _ = torch.min(masked_cluster, dim=1)
            cluster_cost = torch.mean(min_dist_correct) 

            # Separation Cost: Maximize distance to wrong prototype (Minimize negative)
            masked_sep = (cosine_min_distances).masked_fill(prototypes_of_wrong_class == 0, float('-inf'))
            min_dist_wrong, _ = torch.max(masked_sep, dim=1)
            separation_cost = torch.mean(min_dist_wrong)             
            #softmax_scores = F.softmax(project_max_distances, dim=1)
            print('Project max distances', project_max_distances[0,1:20])
            print('Test Output logit', output[0,1:20])

            loss = coefs['crs_ent'] * cross_entropy +  coefs['sep'] * separation_cost +coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep                        
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_subspace_sep_cost += subspace_sep.item()
            total_separation_cost += separation_cost.item()
            total_l1 += l1.item()
            total_loss+= loss.item()
            
            del cosine_min_distances
            del project_max_distances
            del target
            del output
            del predicted
            torch.cuda.empty_cache()
            
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,                       
                    'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'l1':net.last_layer.weight.norm(p=1).item(),
                    'l1_loss' : (total_l1/n_batches),
                    'accu' : n_correct/n_examples,
                    'average loss': (total_loss/ n_batches)}

    return result_loss

def last_only(lr, model):
    """Optimizer specs for Head-Only training"""
    net = model.module if hasattr(model, 'module') else model
    
    # Freeze body
    for p in net.features.parameters(): p.requires_grad = False
    for p in net.add_on_layers.parameters(): p.requires_grad = False
    net.prototype_vectors.requires_grad = False
    
    # Unfreeze head
    for p in net.last_layer.parameters(): p.requires_grad = True
    
    return [{'params': net.last_layer.parameters(), 'lr': lr}]

def beside_last(lrs, model):
    """Optimizer specs for Body-Only training (Features + Prototypes)"""
    net = model.module if hasattr(model, 'module') else model
    
    # Unfreeze body
    for p in net.features.parameters(): p.requires_grad = True
    for p in net.add_on_layers.parameters(): p.requires_grad = True
    net.prototype_vectors.requires_grad = True
    
    # Freeze head
    for p in net.last_layer.parameters(): p.requires_grad = False
    
    # Return list of params for Adam
    return [
        {'params': net.features.parameters(), 'lr': lrs['features'], 'weight_decay': 1e-3},
        {'params': net.add_on_layers.parameters(), 'lr': lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': [net.prototype_vectors], 'lr': lrs['prototype_vectors']}
    ]