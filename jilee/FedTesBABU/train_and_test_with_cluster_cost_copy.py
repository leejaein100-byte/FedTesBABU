import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util.helpers import list_of_distances
from Gr_model_with_cluster_cost import *
from settings_CUB import joint_optimizer_lrs, last_layer_optimizer_lr
from utils.misc import *
from utils.utils import *

def temp_softmax(logits, temp = 1.0):
    """Temperature-scaled softmax (returns Tensor)"""
    return F.softmax(logits / temp, dim=-1)

def get_entropy(logits):
    """Entropy on logits (returns Tensor)"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy  # Tensor

def temp_sharpen_np(x, axis=-1, temp=1.0):
    x = np.maximum(x**(1/temp), 1e-8)
    return x / x.sum(axis=axis, keepdims=True)

def temp_sharpen(x, dim=-1, temp = 1.0, eps= 1e-8):

    x = x.pow(1.0 / temp)
    x = torch.clamp(x, min=eps)
    x = x / x.sum(dim=dim, keepdim=True)

    return x

def get_input_logits(inputs, model, score_logit = True, is_logit = False, device=None):
    """Get logits from model (returns Tensor, no numpy)"""
    model.eval()
    if device is not None:
        inputs = inputs.to(device, non_blocking=True)
    out = model(inputs)
    # 기존 코드가 model(inputs)[-1]을 사용했다면 아래처럼 유지
    if isinstance(out, (list, tuple)):
        if score_logit:
            out = out[-1]
        else:
            out = out[0]
    out = out.detach()
    #entropy = get_entropy(out)
    #print('entropy of teacher models', entropy)
    if not is_logit:
        out = F.softmax(out, dim=1)
    return out  # Tensor (device 상주)
    

def merge_logits(
    logits,  # shape: (B, T, C)
    method= 'mean',
    loss_type = 'KL',
    temp = 0.3):

    B, T, C = logits.shape
    device = logits.device
    dtype = logits.dtype

    if "vote" in method:
        if loss_type == "CE":
            # 다수결 (각 teacher argmax 후 mode)
            votes = logits.argmax(dim=-1)           # (B, T)
            # torch.mode는 dim 지정
            # values: (B,)
            majority = votes.mode(dim=1).values
            # one-hot로 확률 분포화
            logits_arr = F.one_hot(majority, C).to(dtype=dtype)
            logits_cond = logits_arr.max(dim=-1).values  # 항상 1.0
        else:
            # vote + (KL/MSE 등): 평균 후 temp softmax
            mean_logits = logits.mean(dim=1)             # (B, C)
            probs = temp_softmax(mean_logits, temp=temp)
            logits_arr = probs
            logits_cond = probs.max(dim=-1).values
    else:
        mean_logits = logits.mean(dim=1)                 # (B, C)
        if loss_type == "MSE":
            # MSE라면 보통 'logit 평균'을 그대로 쓰기도 하지만
            # 안정적인 KD 파이프라인을 위해 확률도 함께 제공합니다.
            probs = temp_softmax(mean_logits, temp=1.0)
            logits_cond = probs.max(dim=-1).values
        elif "KL" in loss_type:
            #probs = temp_softmax(mean_logits, temp=temp)
            logits_arr = temp_sharpen(mean_logits, temp=temp)
            logits_cond = logits_arr.max(dim=-1).values
        else:
            # 원본 로짓을 그대로 쓰고 싶을 때
            logits_arr = mean_logits
            probs_for_conf = F.softmax(mean_logits, dim=-1)
            logits_cond = probs_for_conf.max(dim=-1).values

    return logits_arr, logits_cond

class BayesianKDTrainer:
    def __init__(self, args, num_classes, net_glob, prototype_per_class = 3, alpha=1.0, temp=0.6, hyperparam= -1e-6):
        self.args = args
        self.num_classes = num_classes
        self.net_glob = net_glob  # Direct reference to global model
        self.alpha = alpha  # Dirichlet concentration parameter
        self.temp = temp    # Temperature for knowledge distillation
        self.prototype_per_class = prototype_per_class
        self.hyperparam = hyperparam
        
    def get_ensemble_logits(
        self, teachers, inputs, method = 'mean', device=None):
        """
        teachers: iterable of nn.Module
        returns logits_arr (KD에 바로 넣을 타깃; 보통 확률 텐서)
        """
        if device is None:
            device = inputs.device
        B = inputs.size(0)
        T = len(teachers)
        logits = torch.zeros(B, T, self.num_classes, device=device)

        for i, teacher in enumerate(teachers):
            t = teacher.module if hasattr(teacher, 'module') else teacher
            t.eval().to(device)
            # 필요시 is_logit=False/True를 맞춰 주세요. 아래는 확률을 받는 예시.
            logit_or_prob = get_input_logits(inputs, t, score_logit =self.args.score_logit, is_logit=False, device=device)  # (B, C)
            #entropy = get_entropy(logit_or_prob)
            logits[:, i, :] = logit_or_prob
            #print('entropy of teacher models', entropy)
        logits_arr, _ = merge_logits(logits, method=method, loss_type='KL', temp=self.temp)
        return logits_arr 


    def create_teacher_models_with_dirichlet(self, clients_state_list, clients, device, num_teachers=30):
        """Create teacher models using Dirichlet distribution from client models"""
        teacher_models = []
        
        # Include original client models as teachers
        for client in clients:
            teacher_models.append(copy.deepcopy(client))
        
        # Create additional ensemble teacher models using Dirichlet distribution
        for teacher_idx in range(num_teachers):
            # Sample proportions from Dirichlet distribution
            #proportions = np.random.dirichlet(np.repeat(self.alpha, len(clients_state_list)))
            consensus_params = np.random.dirichlet(np.repeat(self.alpha, len(clients_state_list)))
            teacher_model = copy.deepcopy(self.net_glob)
            
            # Update non-prototype parameters with Dirichlet-weighted FedAvg
            for k in teacher_model.state_dict().keys():
                if "batches_tracked" in k or "prototype_vectors" in k:   #
                    #print('prototype_vectors detected in BayesianKDTrainer')
                    continue
                
                weighted_param = torch.zeros_like(teacher_model.state_dict()[k])
                for i, client_state in enumerate(clients_state_list):
                    if k in client_state:
                        weighted_param += client_state[k] * consensus_params[i]
                
                teacher_model.state_dict()[k].copy_(weighted_param)
            
            # Update prototype vectors with Dirichlet-parameterized consensus
            if hasattr(teacher_model, 'prototype_vectors'):
                # Collect prototype vectors from all clients
                Gm_manifold = []
                for client_idx, client in enumerate(clients):
                    if hasattr(client, 'prototype_vectors'):
                        prototype_vectors = client.prototype_vectors.data.to(device)
                    elif hasattr(client, 'module') and hasattr(client.module, 'prototype_vectors'):
                        prototype_vectors = client.module.prototype_vectors.data.to(device)
                    else:
                        prototype_vectors = clients_state_list[client_idx]['prototype_vectors'].to(device)
                    
                    Gm_manifold.append(torch.unsqueeze(prototype_vectors, 0))
                
                Gm_manifold = torch.cat(Gm_manifold, dim=0)  # Shape: [num_clients, num_classes, ...]
                if Gm_manifold.dim() == 4:
                    Gm_manifold = Gm_manifold.permute(1, 0, 2, 3)  # Shape: [num_classes, num_clients, ...]
                
                # Apply consensus update for each class using Dirichlet-weighted interactions
                consensus_prototypes = []
                for cls in range(self.num_classes):    #(min(self.num_classes, Gm_manifold.shape[0])):
                    # Get prototype vectors for this class from all clients
                    class_prototypes = Gm_manifold[cls]  # Shape: [num_clients, ...]
                    
                    # Apply consensus algorithm with Dirichlet-parameterized a_{j,k}
                    consensus_prototype = consensus_update_with_dirichlet_weights(
                        class_prototypes, 
                        consensus_params=consensus_params, prototype_per_class= self.prototype_per_class, 
                        iterations=10  # Fixed number of iterations
                    )
                    consensus_prototypes.append(torch.unsqueeze(consensus_prototype, 0))
                
                if consensus_prototypes:
                    teacher_model.prototype_vectors.data = torch.cat(consensus_prototypes, dim=0)
            
            teacher_models.append(teacher_model)

        return teacher_models

    def setup_grassmann_optimizer(self):
        """Setup the specialized Grassmann optimizer for TESNet"""
        
        if not hasattr(self.net_glob, 'module'):   #이게 맞는듯 beside_last는 Multi-gpu에 맞는 세팅이라
            self.net_glob = torch.nn.DataParallel(self.net_glob)

        optimizer_specs = beside_last(joint_optimizer_lrs, self.net_glob)

        # Create Grassmann optimizer
        grassmann_optimizer = GrassmannManifoldOptimizer(
            self.net_glob, optimizer_specs, self.prototype_per_class)
        
        return grassmann_optimizer   #, model
    
    def train_global_model_with_kd(self, teacher_models, server_data, device):
        """Train net_glob using knowledge distillation from teacher models with server data"""
        X_server, y_server = server_data
        
        # Create data loader for server data
        server_dataset = MyDataset(X_server, y_server)
        server_loader = torch.utils.data.DataLoader(
            server_dataset, batch_size=64, shuffle=True
        )
        
        # Prepare global model for training
        self.net_glob.train()
        self.net_glob = self.net_glob.to(device)

        grassmann_optimizer = self.setup_grassmann_optimizer()
        
        total_loss_sum = 0.0
        num_batches = 0
        n_correct1 = 0
        n_correct2 = 0
        n_examples = 0
        for batch_inputs, batch_targets in server_loader:
            #print(f"DEBUG (train_global_model_with_kd): type(batch_inputs) is {type(batch_inputs)}")
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Get ensemble teacher predictions
            with torch.no_grad():
                teacher_logits = self.get_ensemble_logits(teacher_models, batch_inputs)
            
            # Zero gradients
            grassmann_optimizer.zero_grad()
           #_, _, project_max_distances= self.net_glob(batch_inputs)
            if self.args.score_logit:
                _, _, output = self.net_glob(batch_inputs)
            else:
                output, _, _= self.net_glob(batch_inputs)

            #student_logits = F.softmax(project_max_distances, dim=1)
            # Compute knowledge distillation loss
            #total_loss = self.knowledge_distillation_loss(    
                #student_logits, teacher_logits)
            total_loss = self.knowledge_distillation_loss(output, teacher_logits)
            _, predicted1 = torch.max(output, 1)
            _, predicted2 = torch.max(teacher_logits, 1)
            
            n_examples += batch_targets.size(0)
            n_correct1 += (predicted1 == batch_targets).sum().item()
            n_correct2 += (predicted2.to(device) == batch_targets).sum().item()
            # Backward pass
            total_loss.backward()
            
            # Use Grassmann optimizer step
            grassmann_optimizer.step()
            total_loss_sum += total_loss.item()
            num_batches += 1
        
        # Record training results
        avg_total_loss = total_loss_sum / num_batches if num_batches > 0 else 0
        teacher_accu = n_correct2 / n_examples
        stu_accu = n_correct1 / n_examples
        # Unwrap the model if it was wrapped
        if hasattr(self.net_glob, 'module'):
            self.net_glob = self.net_glob.module
        
        return {
            'average loss': avg_total_loss,
            'teacher_accu': teacher_accu, 'stu_accu': stu_accu
        }                
    
    def knowledge_distillation_loss(self, student_score, teacher_logits):
        """Compute knowledge distillation loss"""
        #ce_loss = F.cross_entropy(student_logits, soft_targets)
        #soft_targets = F.softmax(teacher_logits / temperature, dim=1)

    # 2. Get the student's soft log-probability distribution (log(Q))
    # This is the input to the KL divergence loss.
        #soft_pred = F.log_softmax(student_logits, dim=1)
    # 3. Compute the KL divergence loss
    # reduction='batchmean' averages the loss over the batch size (N).
    # The default 'mean' would average over (N * num_classes), which is incorrect.
        #kl_loss = F.kl_div(soft_pred, teacher_logits, reduction='batchmean')
        ce_loss = F.cross_entropy(student_score, teacher_logits)
        projection_operator_1 = torch.unsqueeze(self.net_glob.module.prototype_vectors, dim=0)
        subspace_basis_matrix_T = torch.transpose(self.net_glob.module.prototype_vectors,1,2)
        projection_operator_2 = torch.unsqueeze(subspace_basis_matrix_T, dim = 1)
        del subspace_basis_matrix_T

        projection_operator = torch.matmul(projection_operator_1,projection_operator_2)#[200,128,10] [200,10,128] -> [200,128,128]
        pairwise_distance= torch.norm(projection_operator, dim = [2,3])
        subspace_sep = torch.norm(self.prototype_per_class-pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()
        total_loss = ce_loss + self.hyperparam * subspace_sep
        return total_loss 


def _train_or_test(args, client_idx, client_model, X_data, y_data, body_train, is_train, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):

    client_dataset = MyDataset(X_data, y_data)
    client_model = client_model.train()
    client_model = client_model.cuda()

    if body_train:
        optimizer_specs = beside_last(joint_optimizer_lrs, client_model)
        train_epochs = args.SL_epochs
    else:
        optimizer_specs = last_only(last_layer_optimizer_lr, client_model)
        train_epochs = args.fine_tune_epochs

    optimizer = GrassmannManifoldOptimizer(client_model, optimizer_specs, client_model.module.prototype_per_class) 
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
                output, cosine_max_scores, project_max_distances = client_model(image)   #cosine min distances는 score 점수,즉 무조건 양수
                del image
                if args.score_logit:
                    cross_entropy = torch.nn.functional.cross_entropy(project_max_distances, target)
                else:
                    cross_entropy = torch.nn.functional.cross_entropy(output, target)
                projection_operator_1 = torch.unsqueeze(client_model.module.prototype_vectors, dim=0)
                subspace_basis_matrix_T = torch.transpose(client_model.module.prototype_vectors,1,2)
                projection_operator_2 = torch.unsqueeze(subspace_basis_matrix_T, dim = 1)
                del subspace_basis_matrix_T

                projection_operator = torch.matmul(projection_operator_1,projection_operator_2)#[200,128,10] [200,10,128] -> [200,128,128]
                pairwise_distance= torch.norm(projection_operator, dim = [2,3])
                subspace_sep = torch.norm(client_model.module.prototype_per_class-pairwise_distance, p=1, dim=[0,1], dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()

                del projection_operator_1
                del projection_operator_2
                del pairwise_distance
                #print('Cosine min distances', cosine_max_scores[0,1:20])
                if class_specific:
                    prototypes_of_correct_class = torch.t(client_model.module.prototype_class_identity[:,target]).cuda()  #(N,C)
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    masked = (-cosine_max_scores).masked_fill(prototypes_of_wrong_class == 0, float('-inf'))
                    inverted_distances_to_nontarget_prototypes, _ = torch.max(masked, dim=1)
                    #print('inverted_distances_to_nontarget_prototypes', inverted_distances_to_nontarget_prototypes[0:20])
                    separation_cost = torch.mean(- inverted_distances_to_nontarget_prototypes)
                    inverted_distances, _ = torch.max((cosine_max_scores) * prototypes_of_correct_class, dim=1)
                    #print('inverted_distances', inverted_distances[0:20])
                    cluster_cost = torch.mean(-inverted_distances)
                    #print('Project max distances', project_max_distances[0,1:20])
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(client_model.module.prototype_class_identity).cuda()
                        l1 = (client_model.module.last_layer.weight * l1_mask).norm(p=1)
                    # weight 200,2000   prototype_class_identity [2000,200]
                    else:
                        l1 = client_model.module.last_layer.weight.norm(p=1)

            if is_train:
                if not class_specific:
                    loss = coefs['crs_ent'] * cross_entropy+ coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep
                else:
                    loss = coefs['crs_ent'] * cross_entropy+ coefs['clst'] * cluster_cost+ coefs['sep'] * separation_cost + coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep
                   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #nomalize basis vectors
                #client_model.module.prototype_vectors.data = F.normalize(client_model.module.prototype_vectors, p=2, dim=1).data
            
            del cosine_max_scores, project_max_distances
            
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
                        'average loss': (total_loss/ n_batches)
                        }
    else:
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,
                        'cluster_loss': total_cluster_cost / n_batches,                            
                        'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                        'separation_loss': total_separation_cost / n_batches,
                        'avg_separation': total_avg_separation_cost / n_batches,
                        'l1': client_model.module.last_layer.weight.norm(p=1).item(),
                        'l1_loss' : (total_l1/n_batches),
                        'accu' : n_correct/n_examples,
                        'average loss': (total_loss / n_batches)
                        }
            
    return client_model, result_loss


def local_test_global_model(args, client_model, X_test, y_test, coefs):   #이건 without classifier 버전.
    
    total_cross_entropy = 0
    total_subspace_sep_cost = 0
    total_separation_cost = 0
    total_loss = 0
    total_l1 = 0 
    n_examples = 0
    total_cluster_cost = 0
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
            output, cosine_max_scores, project_max_distances = client_model(image)   #cosine min distances는 score 점수,즉 무조건 양수
            
            del image
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            projection_operator_1 = torch.unsqueeze(client_model.prototype_vectors, dim=0)
            subspace_basis_matrix_T = torch.transpose(client_model.prototype_vectors,1,2)
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
            prototypes_of_correct_class = torch.t(client_model.prototype_class_identity[:,target]).cuda()  #(N,C)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            masked = (-cosine_max_scores).masked_fill(prototypes_of_wrong_class == 0, float('-inf'))
            inverted_distances_to_nontarget_prototypes, _ = torch.max(masked, dim=1)
            separation_cost = torch.mean(- inverted_distances_to_nontarget_prototypes)
            inverted_distances, _ = torch.max((cosine_max_scores) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(-inverted_distances)

            loss = coefs['crs_ent'] * cross_entropy +  coefs['sep'] * separation_cost +coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep +coefs['clst'] * cluster_cost                       
            del cosine_max_scores
            
            #softmax_scores = F.softmax(project_max_distances, dim=1)
            #print('Project max distances', project_max_distances[0,1:20])
            #print('Test Output logit', output[0,1:20])
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
            del project_max_distances
            del target
            del output
            del predicted
            torch.cuda.empty_cache()
            
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'cluster_loss': total_cluster_cost / n_batches,                                                   
                    'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'l1':client_model.last_layer.weight.norm(p=1).item(),
                    'l1_loss' : (total_l1/n_batches),
                    'accu' : n_correct/n_examples,
                    'average loss': (total_loss/ n_batches)}

    return result_loss


def local_test_global_model_proto(args, client_model, X_test, y_test, coefs):   #이건 without classifier 버전.
    
    total_cross_entropy = 0
    total_subspace_sep_cost = 0
    total_separation_cost = 0
    total_loss = 0
    total_l1 = 0 
    n_examples = 0
    total_cluster_cost = 0
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
            output, cosine_max_scores, project_max_distances = client_model(image)   #cosine min distances는 score 점수,즉 무조건 양수
            del image
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            projection_operator_1 = torch.unsqueeze(client_model.prototype_vectors, dim=0)
            subspace_basis_matrix_T = torch.transpose(client_model.prototype_vectors,1,2)
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
            prototypes_of_correct_class = torch.t(client_model.prototype_class_identity[:,target]).cuda()  #(N,C)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            masked = (-cosine_max_scores).masked_fill(prototypes_of_wrong_class == 0, float('-inf'))
            inverted_distances_to_nontarget_prototypes, _ = torch.max(masked, dim=1)
            separation_cost = torch.mean(- inverted_distances_to_nontarget_prototypes)
            inverted_distances, _ = torch.max((cosine_max_scores) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(-inverted_distances)

            loss = coefs['crs_ent'] * cross_entropy +  coefs['sep'] * separation_cost +coefs['l1'] * l1 + coefs['sub_sep'] * subspace_sep +coefs['clst'] * cluster_cost                       
            del cosine_max_scores
            
            #softmax_scores = F.softmax(project_max_distances, dim=1)
            #print('Project max distances', project_max_distances[0,1:20])
            #print('Test Output logit', output[0,1:20])
            _, predicted = torch.max(project_max_distances.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_subspace_sep_cost += subspace_sep.item()
            total_separation_cost += separation_cost.item()
            total_l1 += l1.item()
            total_loss+= loss.item()
            del project_max_distances
            del target
            del output
            del predicted
            torch.cuda.empty_cache()
            
        result_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'cluster_loss': total_cluster_cost / n_batches,                                                   
                    'subspace_sep_loss' : total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'l1':client_model.last_layer.weight.norm(p=1).item(),
                    'l1_loss' : (total_l1 / n_batches),
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
