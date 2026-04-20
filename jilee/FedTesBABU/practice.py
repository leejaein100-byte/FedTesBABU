import torch
import numpy as np
from pymanopt.manifolds import Grassmann
from pymanopt import Problem

def init_Grassmann2(prototype_shape, num_classes, prototype_per_class):
    """
    A numerically stable implementation for Grassmann manifold points.
    
    Args:
        prototype_shape: Tuple with (num_classes, ambient_dimension, ambient_dimension)
                        or [num_classes, ambient_dimension]
        num_classes: Number of classes
        prototype_per_class: Dimension of the subspace (k in G(n,k))
        
    Returns:
        Tensor of shape (num_classes, ambient_dim, ambient_dim) containing projection matrices
    """
    # Extract the ambient dimension
    if isinstance(prototype_shape, tuple) and len(prototype_shape) == 3:
        ambient_dim = prototype_shape[1]
    elif isinstance(prototype_shape, list) and len(prototype_shape) == 2:
        ambient_dim = prototype_shape[1]
    else:
        raise ValueError("prototype_shape should be either (num_classes, dim, dim) or [num_classes, dim]")
    
    # Ensure prototype_per_class is valid
    if prototype_per_class <= 0 or prototype_per_class > ambient_dim:
        raise ValueError(f"prototype_per_class must be between 1 and {ambient_dim}")
    
    # Storage for projection matrices and bases
    projection_matrices = []
    bases_list = []
    
    for _ in range(num_classes):
        # Start with a random matrix
        X = torch.randn(ambient_dim, ambient_dim, dtype=torch.float64)  # Use double precision
        
        # Compute QR decomposition for orthogonal basis
        Q, _ = torch.linalg.qr(X)
        
        # Take first k columns as basis for k-dim subspace
        Y = Q[:, :prototype_per_class]
        
        # Explicitly ensure orthonormality (important for numerical stability)
        if prototype_per_class > 1:
            # Gram-Schmidt process for additional stability
            for j in range(prototype_per_class):
                # Normalize the current column
                col = Y[:, j]
                col = col / torch.norm(col)
                Y[:, j] = col
                
                # Make subsequent columns orthogonal to this one
                for k in range(j+1, prototype_per_class):
                    projection = torch.dot(Y[:, k], col) * col
                    Y[:, k] = Y[:, k] - projection
        else:
            # Just normalize if there's only one vector
            Y = Y / torch.norm(Y, dim=0)
        
        # Store the basis
        bases_list.append(Y)
        
        # Compute projection matrix P = Y Y^T
        P = torch.mm(Y, Y.t())
        
        # Ensure symmetry (explicitly force it)
        P = 0.5 * (P + P.t())
        
        # Convert back to float32 for compatibility
        P = P.to(torch.float64)
        
        # Verify properties - this helps catch any numerical issues early
        is_symmetric = torch.allclose(P, P.t(), rtol=1e-5, atol=1e-5)
        is_idempotent = torch.allclose(P, torch.mm(P, P), rtol=1e-5, atol=1e-5)
        trace = torch.trace(P).item()
        
        if not is_symmetric or not is_idempotent or abs(trace - prototype_per_class) > 1e-4:
            print(f"Warning: Matrix {len(projection_matrices)} has issues:")
            print(f"  Symmetric: {is_symmetric}")
            print(f"  Idempotent: {is_idempotent}")
            print(f"  Trace: {trace} (expected {prototype_per_class})")
            # Try to fix it
            eigenvalues, eigenvectors = torch.linalg.eigh(P)
            # Set eigenvalues to 0 or 1
            fixed_eigenvalues = torch.zeros_like(eigenvalues)
            # The top k eigenvalues should be 1
            fixed_eigenvalues[-prototype_per_class:] = 1.0
            # Reconstruct a valid projection matrix
            P = torch.mm(eigenvectors, torch.mm(torch.diag(fixed_eigenvalues), eigenvectors.t()))
            print("  Matrix has been fixed.")
        
        projection_matrices.append(P)
    
    # Stack all matrices
    projection_matrices = torch.stack(projection_matrices)
    bases = torch.stack(bases_list)
    
    return projection_matrices
    


def init_Grassmann3(prototype_shape, num_classes, prototype_per_class):
    points = []
    basis= torch.nn.init.orthogonal_(torch.empty(prototype_shape[1],prototype_shape[1]))
    
    for i in range(num_classes):
        shuffled_indices = torch.randperm(prototype_shape[1])
        selected_numbers = shuffled_indices[:prototype_per_class]
        pre_eigen = torch.zeros((prototype_shape[1], prototype_shape[1]))
        print('selected_numbers',selected_numbers)
        print('is orthogonal', basis.t()@basis)
        for j in range(prototype_per_class):
            pre_eigen[selected_numbers[j], selected_numbers[j]]=1
        print('pre_eigen', pre_eigen)
        each_class_Grass=basis@pre_eigen@basis.t()
        points.append(each_class_Grass)
    points = torch.stack(points) 

    return points.reshape(num_classes, prototype_shape[1],prototype_shape[1])  #conv2d로 내가 구현하지 않았기 때문에 굳이 4 dimension을 맞출 필요 X

def init_Grassmann(prototype_shape, num_classes, prototype_per_class):
    """
    Properly initialize points on the Grassmann manifold G(n,k) where:
    - n = prototype_shape[1] (ambient dimension)
    - k = prototype_per_class (subspace dimension)
    
    Returns:
        Tensor of shape (num_classes, prototype_shape[1], prototype_shape[1]) containing
        projection matrices representing points on the Grassmann manifold.
    """
    ambient_dim = prototype_shape[1]
    subspace_dim = prototype_per_class
    manifold = Grassmann(ambient_dim, subspace_dim)
    projection_matrices = []
    for _ in range(num_classes):
        # Sample a random point (basis) from the manifold
        Y = manifold.random_point()  # Shape: (ambient_dim, subspace_dim)
        Y_torch = torch.tensor(Y, dtype=torch.float32)
        #Y_torch = torch.tensor(Y, dtype=torch.float32)
        
        # Form projection matrix P = Y*Y^T
        P = Y_torch @ Y_torch.t()
        print(Y_torch.t()@Y_torch)
        projection_matrices.append(P)
    
    # Stack all projection matrices
    projection_matrices = torch.stack(projection_matrices)

    return projection_matrices



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
    norm_prototype_vectors = torch.nn.functional.normalize(prototype_vectors, p=2, dim=2)
    print('norm_prototype_vectors size', norm_prototype_vectors.size())
    _, Ch, _ = norm_prototype_vectors.size()
    
    # Reshape to get the matrices
    #matrices = norm_prototype_vectors.reshape(num_classes, Ch, Ch)
    
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
        P = norm_prototype_vectors[i]
        
        # 1. Check symmetry: P = P^T
        is_symmetric = torch.allclose(P, P.transpose(0, 1), rtol=1e-5, atol=1e-5)
        
        # 2. Check idempotence: P^2 = P
        P_squared = torch.matmul(P, P)
        is_idempotent = torch.allclose(P, P_squared, rtol=1e-5, atol=1e-5)
        
        # 3. Get eigenvalues (for a projection matrix, eigenvalues should be 0 or 1)
        eigenvalues = torch.linalg.eigvalsh(P)
        
        # 4. Calculate rank (number of non-zero eigenvalues, should equal trace for projection matrices)
        rank = sum((eigenvalues > 0.3).float())
        
        # 5. Calculate trace (equals rank for projection matrices)
        trace = torch.trace(P)
        
        # Store results
        results['is_symmetric'].append(is_symmetric)
        results['is_idempotent'].append(is_idempotent)
        results['eigenvalues'].append(eigenvalues)
        results['rank'].append(rank)
        results['trace'].append(trace)
    
    # Overall summary
    results['all_symmetric'] = all(results['is_symmetric'])
    results['all_idempotent'] = all(results['is_idempotent'])
    results['summary'] = f"Matrices checked: {num_classes}\n"
    results['summary'] += f"All symmetric: {results['all_symmetric']}\n"
    results['summary'] += f"All idempotent: {results['all_idempotent']}\n"
    
    if not results['all_symmetric'] or not results['all_idempotent']:
        results['summary'] += "ERROR: Not all matrices are proper projection matrices!\n"
    else:
        results['summary'] += "SUCCESS: All matrices are proper projection matrices.\n"
    
    return results


points = init_Grassmann2(prototype_shape=(200,32,32), num_classes=200, prototype_per_class=3)
result = verify_projection_matrices(points, num_classes=200)
print('Verify projection matrices', result['trace'])