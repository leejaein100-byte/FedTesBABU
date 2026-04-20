import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent
import autograd.numpy as anp
from torch.utils.data import Dataset, DataLoader
from models.resnet_features import *
from models.densenet_features import *
from models.vgg_features import *
import copy
#import scipy.linalg as la

def consensus_update_with_dirichlet_weights_projector_level(prototype_vectors, consensus_params, prototype_per_class, iterations=30, alpha=0.1):
    """
    Consensus update algorithm based on equation (16) from the paper for Grassmann manifolds:
    d/dt Π_k = 2α ∑_j a_{jk} (Π_k Π_j Π_k^⊥ + Π_k^⊥ Π_j Π_k)
    
    where Π_k are projection matrices (prototype vectors) and Π_k^⊥ = I - Π_k
    
    Args:
        prototype_vectors: Tensor of shape [num_clients, num_classes, feature_dim, feature_dim] 
                        - projection matrices from all clients
        prototype_vectors: [num_clients, feature_dim, feature_dim]
        consensus_params: Array of length num_clients - Dirichlet parameters for a_{j,k}
        iterations: Number of consensus iterations
        alpha: Step size for consensus update
    
    Returns:
        consensus_prototype: Final consensus projection matrix
    """
    num_clients = prototype_vectors.shape[0]
    current_projections = prototype_vectors.clone()  # Shape: [num_clients, feature_dim, feature_dim]
    
    # Get identity matrix for computing complements
    I = torch.eye(current_projections.shape[-1], device=current_projections.device)
    for iteration in range(iterations):

        if iteration ==0:
            Π_k = current_projections[0]
        else:
            Π_k = updated_projection
            # Current projection matrix for client k
        Π_k_perp = I - Π_k  # Complement: Π_k^⊥ = I - Π_k
        
        # Compute consensus term: ∑_j a_{jk} (Π_k Π_j Π_k^⊥ + Π_k^⊥ Π_j Π_k)
        consensus_term = torch.zeros_like(Π_k)
        
        for j in range(num_clients):
    
            Π_j = current_projections[j]  # Projection matrix from client j
            a_jk = consensus_params[j]  # Dirichlet-sampled weight
            
            # Equation (16): a_{jk} (Π_k Π_j Π_k^⊥ + Π_k^⊥ Π_j Π_k)
            term1 = torch.mm(torch.mm(Π_k, Π_j), Π_k_perp)  # Π_k Π_j Π_k^⊥
            term2 = torch.mm(torch.mm(Π_k_perp, Π_j), Π_k)  # Π_k^⊥ Π_j Π_k
            
            consensus_term += a_jk * (term1 + term2)
    
        # Update: Π_k = Π_k + 2α * consensus_term
        updated_projection = Π_k + 2 * alpha * consensus_term
        updated_projection = project_to_projection_matrix(updated_projection,prototype_per_class)            
    
    return updated_projection

def consensus_update_with_dirichlet_weights(prototype_vectors, consensus_params, prototype_per_class, iterations=30, alpha=0.1, mode='polar'):
    """
    Modified consensus update utilizing basis-level retractions and Dirichlet weights.
    Reflects the logic from consensus_update.txt while using the weights from the first file.
    """
    num_clients = prototype_vectors.shape[0]
    feature_dim = prototype_vectors.shape[-1]
    p = prototype_per_class

    # 1. Initialize by extracting basis vectors for all clients
    # This reflects the basis_group initialization from the second file [cite: 498]
    basis_group = []
    for k in range(num_clients):
        # Assumes get_basis_from_projector is available in your environment
        basis_group.append(get_basis_from_projector(prototype_vectors[k], p))
    
    # Starting basis (Y_k) is taken from the first client [cite: 498]
    Y_k = basis_group[0]
    I = torch.eye(feature_dim, device=Y_k.device)

    for iteration in range(iterations):
        # 2. Compute current projection and its complement [cite: 500]
        Pi_k = Y_k @ Y_k.t()
        Pi_k_perp = I - Pi_k
        
        # 3. Compute Tangent Direction (Xi_k) incorporating Dirichlet weights
        # We replace the simple sum with a weighted sum using consensus_params [cite: 496, 501, 502]
        weighted_Pi_sum = torch.zeros((feature_dim, feature_dim), device=Y_k.device)
        for j in range(num_clients):
            a_jk = consensus_params[j] # Reflect Dirichlet weights [cite: 495]
            weighted_Pi_sum += a_jk * prototype_vectors[j]
        
        # Project the weighted sum onto the tangent space [cite: 502]
        Xi_k = Pi_k_perp @ weighted_Pi_sum @ Y_k        

        # 4. Apply Basis-Level Retraction 
        if mode == 'qr':
            # QR Retraction: qf(Y + alpha * Xi) [cite: 502]
            Y_k = torch.linalg.qr(Y_k + alpha * Xi_k).Q
        elif mode == 'polar':
            # Polar Retraction using SVD 
            U, _, Vh = torch.linalg.svd(Y_k + alpha * Xi_k, full_matrices=False)
            Y_k = U @ Vh
            
    # 5. Return as a projector for architecture compatibility 
    return Y_k @ Y_k.t()


def FedAvg(w_list, glob_state_dict):
    try:
        w_avg = copy.deepcopy(glob_state_dict)

        for k in w_avg.keys():
            # Skip prototype_vectors and last_layer parameters
            #print('parameters of global model', k)
            if 'prototype_vectors' in k or 'last_layer' in k:
                #print('1')
                continue
                
            # Initialize with zeros of the same shape and type
            w_avg[k] = torch.zeros_like(w_list[0][k], dtype=torch.float)
            
            # Sum parameters from all client models
            for i in range(len(w_list)):
                w_avg[k] += w_list[i][k].float()
                
            # Average by dividing by the number of clients
            w_avg[k] = torch.div(w_avg[k], len(w_list))
            
        return w_avg
        
    except Exception as e:
        print(f"Error in FedAvg: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def frechet_mean(points, max_iterations=10, tolerance=1e-3):
    """Compute the Fréchet mean of points on the Grassmann manifold."""
    # Initialize Fréchet mean as the initial guess (e.g., first point)
    mean = points[0]
    device = mean.device
    mean = mean.to(device)
    
    for point in points:
        point = point.to(device)
    
    for _ in range(max_iterations):
        # Compute the geodesic distances to the current mean
        #distances = [geodesic_distance(mean, point) for point in points]
        
        #weighted_sum = np.zeros_like(mean)
        temp = torch.zeros_like(mean)
        for point in points:
            temp += grassmann_log_map(point, mean)
            #temp += torch.(point)
        temp= temp/len(points)
        new_mean = grassmann_exp_map(mean, temp)
        #print('mean computation', new_mean)
        # Compute the new mean
        #new_mean = weighted_sum / total_weight
        
        # Check for convergence
        if torch.linalg.norm(mean - new_mean) < tolerance:
            break
        
        mean = new_mean
    
    return mean
    
def compute_orthogonal_complement(Pi):
    """Compute orthogonal complement of projection matrix Pi"""
    if Pi.dim() == 2:
        # For 2D matrices
        I = torch.eye(Pi.shape[0], device=Pi.device, dtype=Pi.dtype)
        return I - Pi
    elif Pi.dim() >= 3:
        # For higher dimensional tensors, compute complement along last two dimensions
        *batch_dims, h, w = Pi.shape
        I = torch.eye(h, device=Pi.device, dtype=Pi.dtype)
        I = I.expand(*batch_dims, h, w)
        return I - Pi
    else:
        raise ValueError(f"Unsupported tensor dimension: {Pi.dim()}")

def consensus_update_without_basis_conversion(Pi_list, rank=3, alpha=0.01, iterations=200):
    """
    Perform iterative consensus updates on projection matrices Pi using the given adjacency matrix.
    Pi_list: List of projection matrices from different clients.
    adjacency_matrix: NxN matrix defining communication weights.
    alpha: Step size for consensus updates.
    iterations: Number of consensus iterations.
    """
    N = len(Pi_list)  # Number of clients
    for iter in range(iterations):
        if iter ==0:
            Pi_k = Pi_list[0]
        else:
            Pi_k = new_Pi_k
        Pi_k_perp = compute_orthogonal_complement(Pi_k)
        update = torch.zeros_like(Pi_k)

        for j in range(N): 
            Pi_j = Pi_list[j]
            term1 = Pi_k @ Pi_j @ Pi_k_perp
            term2 = Pi_k_perp @ Pi_j @ Pi_k
            update += (term1 + term2)

        # Apply the consensus update
        new_Pi_k = Pi_k + 2 * alpha * update

        new_Pi_k = project_to_projection_matrix(new_Pi_k,rank)
        #new_Pi_k = Pi_k  # Update all clients
    return new_Pi_k

def get_basis_from_projector(Pi, p):
    """Extracts the p-dimensional orthonormal basis from a projection matrix."""
    # Pi = Y Y^T, so we take the top p eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(Pi)
    # eigh returns ascending; take the last p
    return eigenvectors[:, -p:]

def consensus_update(Pi_list, p=3, alpha=0.1, iterations=200, mode='polar'):    #with basis, qr contraction 쓰는게 맞나??
    """
    Performs basis-level consensus updates.
    Pi_list: List of projection matrices from clients.
    p: Dimension of the manifold (rank of projection matrix).
    """
    N = len(Pi_list)
    # Initialize basis from the first client's projector
    #Y_k = get_basis_from_projector(Pi_list[0], p)
    basis_group = []
    for k in range(N):
        basis_group.append(get_basis_from_projector(Pi_list[k],p))
    Y_k = basis_group[0]
    # Pre-convert all client projectors to bases for consistency (optional)
    # or keep them as projectors to use Eq. (tangent-direction)
    
    for iter in range(iterations):
        # 1. Compute Tangent Direction (Xi_k)
        # Using Eq: Xi_k = sum_j a_jk * (I - Y_k Y_k^T) * Y_j
        I = torch.eye(Y_k.shape[0], device=Y_k.device)
        Pi_k = Y_k @ Y_k.t()
        Pi_k_perp = I - Pi_k
        
        Xi_k = torch.zeros_like(Pi_list[0])
        for j in range(N):

            Xi_k += Pi_list[j]
        Xi_k = Pi_k_perp @ Xi_k @ Y_k        
        # 2. Apply Retraction
        if mode == 'qr':
            # QR Retraction: qf(Y + alpha * Xi)
            Y_k = torch.linalg.qr(Y_k + alpha * Xi_k).Q
        elif mode == 'polar':
            # Polar Retraction using SVD for stability
            U, _, Vh = torch.linalg.svd(Y_k + alpha * Xi_k, full_matrices=False)
            Y_k = U @ Vh
            
    # Return as a projector for architecture compatibility
    return Y_k @ Y_k.t()


def St_frechet_mean(points, max_iterations=100):
    """
    Compute the Fréchet mean of points on a product of Stiefel manifolds.
    
    This function finds a single "mean" point, X, of shape (num_class, dim, C),
    that minimizes the sum of distances to all N user points.
    
    Parameters:
    points (torch.Tensor): A tensor of shape (num_class, num_users, dim, C)
    max_iterations (int): Max iterations for the solver.
    
    Returns:
    torch.Tensor: Fréchet mean, shape (num_class, dim, C)
    float: Final cost value
    """
    
    # 1. Get dimensions from the input tensor
    num_users, num_class, dim, C = points.shape
    
    # 3. Define the product manifold St(dim, C)^num_class
    # n (rows) = dim
    # p (columns) = C
    # k (product manifold size) = num_class
    manifold = Stiefel(n=dim, p=C, k=num_class)

    # 4. Define the cost function
    # This cost function will be optimized by Pymanopt.
    @pymanopt.function.pytorch(manifold)
    def cost(X):
        # 'X' is the current estimate for the mean, shape (num_class, dim, C)
        total_cost = 0.0
        
        # We iterate over all 'num_users' points
        for i in range(num_users):

            point_i = points[i].to(X.device) 
            
            # Calculate the squared chordal distance (Frobenius norm of differences)
            # This is a common objective for the Fréchet mean on Stiefel.
            
            # X_T_X shape: (num_class, C, C)
            #X_T_X = torch.bmm(X.transpose(1, 2), X)
            X_T_X = torch.bmm(X, X.transpose(1, 2))
            
            # P_T_P shape: (num_class, C, C)
            #P_T_P = torch.bmm(point_i.transpose(1, 2), point_i)
            P_T_P = torch.bmm(point_i, point_i.transpose(1, 2))
            
            # diff shape: (num_class, C, C)
            diff = X_T_X - P_T_P
            
            # We sum the Frobenius norm over all 'num_class' components.
            # torch.linalg.norm(diff, ord='fro') computes the total
            # Frobenius norm summed over the batch dimension (num_class).
            total_cost += torch.linalg.norm(diff, ord='fro', dim=(1,2)).mean()
            
        return total_cost
    
    # 5. Create and solve the optimization problem
    problem = Problem(manifold=manifold, cost=cost)
    
    # SteepestDescent is usually a stable choice
    solver = SteepestDescent(max_iterations=max_iterations)
    
    # 6. Use the first user's point as the initial guess
    # Pymanopt's solver expects a numpy array as the initial point
    Xinit = points[0].cpu().numpy() if isinstance(points[0], torch.Tensor) else points[0]    
    # 7. Run the solver
    mean_point_np = solver.run(problem, initial_point=Xinit).point
    
    # 8. Return the result as a torch tensor
    return torch.from_numpy(mean_point_np)

def stiefel_consensus_update(Y_list, alpha=0.1, iterations=100, mode='polar'):
    """
    Performs consensus updates specifically for the Stiefel Manifold.
    Y_list: List of [n, p] orthogonal matrices (client prototypes).
    alpha: Step size (learning rate for the mean search).
    iterations: Number of consensus steps.
    """
    N = len(Y_list)
    # Initialize with the average in Euclidean space (needs retraction immediately)
    #Y_avg = torch.stack(Y_list).mean(dim=0)
    Y_avg = Y_list[0]
    # Initial Retraction to ensure we start on the manifold
    U, _, Vh = torch.linalg.svd(Y_avg, full_matrices=False)
    Y_k = U @ Vh

    for i in range(iterations):
        # 1. Compute the Stiefel Gradient (Tangent Direction)
        # For the Stiefel manifold, the gradient of the distance-sum is:
        # G = sum( (I - YY^T)Y_j - Y(Y_j^T Y - Y^T Y_j)/2 )
        
        I = torch.eye(Y_k.shape[0], device=Y_k.device)
        Xi_k = torch.zeros_like(Y_k)
        
        for j in range(N):
            Y_j = Y_list[j]
            # Projection onto Tangent Space
            # Part A: ensures orthogonality to the current subspace
            part_a = (I - Y_k @ Y_k.t()) @ Y_j
            # Part B: ensures the internal rotation is averaged
            #part_b = Y_k @ (torch.linalg.solve(Y_k.t() @ Y_k, Y_j.t() @ Y_k - Y_k.t() @ Y_j)) / 2
            part_b = Y_k @ (Y_k.T @ Y_j - Y_j.T @ Y_k) / 2
            Xi_k += (part_a + part_b)
        
        # 2. Update and Retract
        # Move in the direction of the average gradient
        Y_next = Y_k + (alpha / N) * Xi_k
        
        if mode == 'polar':
            # Polar retraction (SVD-based) is most stable for Stiefel 
            U, _, Vh = torch.linalg.svd(Y_next, full_matrices=False)
            Y_k = U @ Vh
        elif mode == 'qr':
            # QR is faster but less 'centered' than Polar
            Q, R = torch.linalg.qr(Y_next)
            # Ensure deterministic QR by fixing sign of R diagonal
            d = torch.diag(R).sign()
            Y_k = Q * d
            
        # Optional: Check for convergence
        if torch.norm((alpha / N) * Xi_k) < 1e-7:
            break
            
    return Y_k

def verify_projection_matrices(prototype_vectors, num_classes):
    """
    Verify if the reshaped prototype vectors form proper projection matrices.
    
    Args:
        prototype_vectors: The prototype vectors tensor to verify
        num_classes: Number of classes
        
    Returns:
        Dictionary containing verification results
    """
    # Normalize the prototype vectors (as in your original code)
    #norm_prototype_vectors = torch.nn.functional.normalize(prototype_vectors, p=2, dim=2)
    
    # Get the dimensions
    _, Ch, _ = prototype_vectors.shape
    
    # Reshape to get the matrices
    matrices = prototype_vectors.reshape(num_classes, Ch, Ch)
    
    # Results dictionary
    results = {
        'is_symmetric': [],
        'is_idempotent': [],
        'eigenvalues': [],
        'rank': [],
        'trace': []
    }
    
    # Check each matrix
    for i in range(num_classes):
        P = matrices[i]
        
        # 1. Check symmetry: P = P^T
        is_symmetric = torch.allclose(P, P.transpose(0, 1), rtol=1e-1, atol=1e-1)
        
        # 2. Check idempotence: P^2 = P
        P_squared = torch.matmul(P, P)
        is_idempotent = torch.allclose(P, P_squared, rtol=1e-2, atol=5e-1)
        
        # 3. Get eigenvalues (for a projection matrix, eigenvalues should be 0 or 1)
        eigenvalues = torch.linalg.eigvalsh(P)
        
        # 4. Calculate rank (number of non-zero eigenvalues, should equal trace for projection matrices)
        rank = sum((eigenvalues > 0.01).float())
        
        # 5. Calculate trace (equals rank for projection matrices)
        trace = torch.trace(P)
        
        # Store results
        results['is_symmetric'].append(is_symmetric)
        results['is_idempotent'].append(is_idempotent)
        results['eigenvalues'].append(eigenvalues)
        results['rank'].append(rank)
        results['trace'].append(trace)
    
    # Overall summary
    results['all_symmetric'] = sum(results['is_symmetric'])
    results['all_idempotent'] = sum(results['is_idempotent'])
    results['summary'] = f"Matrices checked: {num_classes}\n"
    results['summary'] += f"All symmetric: {results['all_symmetric']}\n"
    results['summary'] += f"All idempotent: {results['all_idempotent']}\n"
    results['summary'] += f"All rank: {results['rank']}\n"
    
    if not results['all_symmetric'] or not results['all_idempotent']:
        results['summary'] += "ERROR: Not all matrices are proper projection matrices!\n"
    else:
        results['summary'] += "SUCCESS: All matrices are proper projection matrices.\n"
    
    return results

def project_to_projection_matrix_first(matrix, target_rank=3):
    """
    Alternative implementation that explicitly maintains a target rank.
    
    Args:
        matrix: Input matrix to project
        target_rank: Desired rank of the projection matrix (number of 1s in eigenvalues)
    """
    # Ensure symmetry
    symmetric_matrix = (matrix + matrix.t()) / 2
    
    # Eigendecomposition (for symmetric matrices, SVD = eigendecomposition)
    #eigenvalues, eigenvectors = torch.symeig(symmetric_matrix, eigenvectors=True)
    #try:
    #    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix, UPLO='L')
    #except torch._C._LinAlgError:
        # Fallback to SVD which is more numerically stable
    #    print("Warning: Using SVD fallback due to eigendecomposition failure")
    U, S, Vh = torch.linalg.svd(matrix)
    # Use SVD results as eigendecomposition approximation
    eigenvalues = S
    eigenvectors = U
# Sort eigenvalues in descending order
    sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Create projected eigenvalues: top target_rank become 1, rest become 0
    projected_eigenvalues = torch.zeros_like(sorted_eigenvalues)
    if target_rank > 0:
        projected_eigenvalues[:min(target_rank, len(projected_eigenvalues))] = 1.0
    
    # Reconstruct the projection matrix
    projected_matrix = torch.mm(
        torch.mm(sorted_eigenvectors, torch.diag(projected_eigenvalues)), 
        sorted_eigenvectors.t()
    )
    
    return projected_matrix

def project_to_projection_matrix(matrix, target_rank=3):
    """
    Projects a matrix to a projection matrix using eigenvalue decomposition.
    
    Args:
        matrix: Input matrix to project
        target_rank: Desired rank of the projection matrix (number of 1s in eigenvalues)
    """
    # Ensure symmetry
    symmetric_matrix = (matrix + matrix.t()) / 2

    # Perform eigenvalue decomposition on the symmetric matrix
    # eigh returns eigenvalues in ascending order by default.
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
    
    # Sort eigenvalues in descending order
    sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Create projected eigenvalues: top target_rank become 1, rest become 0
    projected_eigenvalues = torch.zeros_like(sorted_eigenvalues)
    if target_rank > 0:
        projected_eigenvalues[:min(target_rank, len(projected_eigenvalues))] = 1.0
    
    # Reconstruct the projection matrix
    # P = V * D_proj * V^T
    projected_matrix = torch.mm(
        torch.mm(sorted_eigenvectors, torch.diag(projected_eigenvalues)), 
        sorted_eigenvectors.t()
    )
    
    return projected_matrix