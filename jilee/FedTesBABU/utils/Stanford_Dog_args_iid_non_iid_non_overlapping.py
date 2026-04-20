import os
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import math
from collections import defaultdict
from util.preprocess import mean, std

def distributed_setup_seed(seed):
    """Utility to reset all seeds globally for a specific trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def distribute_cub_iid_class_balanced(y, num_users, server_id_size=0, seed=42):
    """
    Distribute CUB samples in a class-balanced IID manner.
    Every user gets a nearly equal slice of every class.
    
    Args:
        y (torch.Tensor): Labels tensor
        num_users (int): Number of federated users
        server_id_size (int): Total samples reserved for server (balanced across classes)
        seed (int): Random seed
        
    Returns:
        tuple: (dict_users, server_indices)
    """
    np.random.seed(seed)
    n_samples = len(y)
    K = int(torch.max(y).item()) + 1
    all_indices_set = set(range(n_samples))
    y_np = y.cpu().numpy()
    
    # 1. Balanced Server Site Allocation (Similar to Stanford Dogs balanced style)
    server_indices = []
    if server_id_size > 0:
        samples_per_class_server = max(1, server_id_size // K)
        for k in range(K):
            idx_k = np.where(y_np == k)[0]
            if len(idx_k) >= samples_per_class_server:
                selected = np.random.choice(idx_k, samples_per_class_server, replace=False)
                server_indices.extend(selected.tolist())
    
    # 2. Define pool for users
    local_indices = list(all_indices_set - set(server_indices))
    local_index_set = set(local_indices)
    
    # 3. Class-balanced IID assignment using np.array_split
    dict_users = {i: [] for i in range(num_users)}
    for k in range(K):
        # Get indices for class k that are NOT in the server set
        idx_k = np.where(y_np == k)[0]
        idx_k = [idx for idx in idx_k if idx in local_index_set]
        np.random.shuffle(idx_k)
        
        # Split this class as evenly as possible across all users
        parts = np.array_split(idx_k, num_users)
        for user_idx, part in enumerate(parts):
            if len(part) > 0:
                dict_users[user_idx].extend(part.tolist())
    
    # Final shuffle for each user
    for user_idx in range(num_users):
        np.random.shuffle(dict_users[user_idx])
        
    return dict_users, server_indices

def distribute_stanford_dogs_iid(total_samples, num_users, server_id_size=0, seed=42):
    """
    Distribute Stanford Dogs samples in an IID manner among users.
    
    Args:
        total_samples (int): Total number of samples
        num_users (int): Number of federated users
        server_id_size (int): Number of samples reserved for server
        seed (int): Random seed
        
    Returns:
        tuple: (dict_users, server_indices)
    """
    np.random.seed(seed)
    all_indices = list(range(total_samples))
    np.random.shuffle(all_indices)
    
    # Reserve server data first
    server_indices = []
    if server_id_size > 0:
        server_indices = all_indices[:server_id_size]
        all_indices = all_indices[server_id_size:]
    
    # Distribute remaining samples equally among users
    samples_per_user = len(all_indices) // num_users
    dict_users = {}
    
    for user_id in range(num_users):
        start_idx = user_id * samples_per_user
        if user_id == num_users - 1:  # Last user gets remaining samples
            end_idx = len(all_indices)
        else:
            end_idx = (user_id + 1) * samples_per_user
        dict_users[user_id] = all_indices[start_idx:end_idx]
    
    return dict_users, server_indices

def distribute_stanford_dogs_dirichlet(X, y, num_users, alpha=0.5, server_id_size=0, tr_frac=0.8, seed=42, min_per_label=5):
    """
    Distribute Stanford Dogs data indices in a non-IID manner following a Dirichlet distribution.
    Ensures that each user's train and test splits have consistent distribution.
    
    Args:
        X (torch.Tensor): Data tensor
        y (torch.Tensor): Labels tensor
        num_users (int): Number of users to distribute data to
        alpha (float): Concentration parameter of the Dirichlet distribution
                     Lower alpha -> more heterogeneous distribution (more non-IID)
                     Higher alpha -> more homogeneous distribution (closer to IID)
        server_id_size (int): Number of samples for the server
        tr_frac (float): Fraction of data to use for training (per user)
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary mapping user IDs to their training data indices
        dict: Dictionary mapping user IDs to their test data indices
        list: Server training data indices (if server_id_size > 0)
        list: Server test data indices (if server_id_size > 0)
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Get total number of samples and classes
    n_samples = len(y)
    n_classes = len(torch.unique(y))

    print(f"Distributing {n_samples} samples among {num_users} users using Dirichlet distribution (alpha={alpha})")
    print(f"Dataset has {n_classes} classes")
    print(f"Target minimum per-user samples per label: {min_per_label}")

    # Dictionary to store user indices (before train/test split)
    dict_users_all = {i: [] for i in range(num_users)}

    # Server indices (before train/test split)
    server_indices_all = []

    # Create index list for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label.item()].append(idx)

    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])

    print(f"Class distribution in original dataset:")
    for class_idx in sorted(class_indices.keys()):
        print(f"  Class {class_idx}: {len(class_indices[class_idx])} samples")

    # ------------------------------------------------------------------
    # Step 1) FedBE-style anchor allocation:
    # give each user at least `min_per_label` per class when possible.
    # ------------------------------------------------------------------
    effective_min_per_label = {}
    for class_idx in sorted(class_indices.keys()):
        available = len(class_indices[class_idx])
        per_user_min = min(min_per_label, available // num_users)
        effective_min_per_label[class_idx] = per_user_min
        if per_user_min < min_per_label:
            print(
                f"[WARN] Class {class_idx}: requested min {min_per_label}, "
                f"but only {available} samples available. Using {per_user_min} per user."
            )

        take = per_user_min * num_users
        if take == 0:
            continue

        anchor_indices = class_indices[class_idx][:take]
        class_indices[class_idx] = class_indices[class_idx][take:]
        anchor_indices = np.array(anchor_indices).reshape(num_users, per_user_min)

        for user_idx in range(num_users):
            dict_users_all[user_idx].extend(anchor_indices[user_idx].tolist())

    # ------------------------------------------------------------------
    # Step 2) Optional balanced server allocation from remaining samples.
    # ------------------------------------------------------------------
    if server_id_size > 0:
        samples_per_class = max(1, server_id_size // n_classes)
        print(f"Allocating up to {samples_per_class} samples per class to server")

        for class_idx in sorted(class_indices.keys()):
            indices = class_indices[class_idx]
            take = min(samples_per_class, len(indices))
            if take == 0:
                continue
            selected_indices = indices[:take]
            server_indices_all.extend(selected_indices)
            class_indices[class_idx] = indices[take:]

    # ------------------------------------------------------------------
    # Step 3) Dirichlet allocation for remaining samples.
    # ------------------------------------------------------------------
    for class_idx, indices in class_indices.items():
        if not indices:
            continue

        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        cut_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(np.array(indices), cut_points)

        for user_idx, part in enumerate(split_indices):
            if len(part) > 0:
                dict_users_all[user_idx].extend(part.tolist())
    
    # Now split each user's data into train and test WHILE maintaining class distribution
    dict_users_train = {i: [] for i in range(num_users)}
    dict_users_test = {i: [] for i in range(num_users)}
    
    for user_idx, indices in dict_users_all.items():
        if not indices:
            continue
            
        # Group user's data by class
        user_class_indices = defaultdict(list)
        for idx in indices:
            label = y[idx].item()
            user_class_indices[label].append(idx)
        
        # For each class, split into train and test with the same proportion
        for class_label, class_indices_user in user_class_indices.items():
            n_train = round(len(class_indices_user) * tr_frac)
            np.random.shuffle(class_indices_user)
            train_indices = class_indices_user[:n_train]
            test_indices = class_indices_user[n_train:]
            
            # Append to user's train and test sets
            dict_users_train[user_idx].extend(train_indices)
            dict_users_test[user_idx].extend(test_indices)
    
    # Split server data into train and test
    #server_train_indices = []
    #server_test_indices = []
    server_indices = []
    if server_id_size > 0:
        # Group server data by class
        server_class_indices = defaultdict(list)
        for idx in server_indices_all:
            label = y[idx].item()
            server_class_indices[label].append(idx)
        
        # For each class, split into train and test
        for class_label, indices in server_class_indices.items():
            np.random.shuffle(indices)
            server_indices.extend(indices)
    # Print distribution statistics
    print(f"\nNon-IID Distribution Statistics (alpha={alpha}):")
    #print(f"Server: {len(server_train_indices)} train, {len(server_test_indices)} test samples")
    
    for user_idx in range(num_users):
        train_indices = dict_users_train[user_idx]
        test_indices = dict_users_test[user_idx]
        
        if not train_indices:
            print(f"User {user_idx}: No data")
            continue
        
        train_labels = [y[idx].item() for idx in train_indices]
        test_labels = [y[idx].item() for idx in test_indices]
        
        train_class_counts = defaultdict(int)
        test_class_counts = defaultdict(int)
        
        for label in train_labels:
            train_class_counts[label] += 1
        
        for label in test_labels:
            test_class_counts[label] += 1
        
        unique_train_classes = len(train_class_counts)
        unique_test_classes = len(test_class_counts)
        
        print(f"User {user_idx}: {len(train_indices)} train ({unique_train_classes} classes), {len(test_indices)} test ({unique_test_classes} classes)")
        
        # Show top 3 classes for train and test to verify consistency
        if len(train_class_counts) > 0:
            train_top = sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            test_top = sorted(test_class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  Train top classes: {[f'Class {c}: {n}' for c, n in train_top]}")
            print(f"  Test top classes:  {[f'Class {c}: {n}' for c, n in test_top]}")
    
    # Final check: each user should have at least `effective_min_per_label[class]`
    # samples per label across local (train+test) data.
    for user_idx in range(num_users):
        user_all_indices = dict_users_train[user_idx] + dict_users_test[user_idx]
        if not user_all_indices:
            continue
        user_labels = [y[idx].item() for idx in user_all_indices]
        label_counts = defaultdict(int)
        for label in user_labels:
            label_counts[label] += 1
        for class_idx in sorted(class_indices.keys()):
            required = effective_min_per_label[class_idx]
            if required > 0 and label_counts[class_idx] < required:
                print(
                    f"[WARN] User {user_idx}, class {class_idx}: "
                    f"{label_counts[class_idx]} < required {required}"
                )

    return dict_users_train, dict_users_test, server_indices 
    


def _load_StanfordDogs(args):
    """
    Load a mixed dataset of Stanford Dogs.
    
    Args:
        data_size (int): Total number of samples to include in the mixed dataset
        root_path (str): Path to the Stanford Dogs directory
        use_bbox (bool): True = use cropped images, False = use original images
        
    Returns:
        tuple: (X, y) tensors containing images and labels
    """

    normalize = transforms.Normalize(mean=mean,std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize])
    # Create custom dataset
    image_type = "cropped" if args.use_bbox else "original"
    print(f"Loading Stanford Dogs dataset ({image_type} images)...")
    
    try:
        stanford_dogs_dataset = StanfordDogsDataset(args, transform)
    except Exception as e:
        print(f"Error loading Stanford Dogs dataset: {e}")
        if args.use_bbox:
            print("Please ensure cropped images are available at /home/jilee/jilee/data/Cropped Images/")
            print("Or set use_bbox=False to use original images")
        else:
            print("Please ensure original images are available at /root/Tesla/integrated/integrated/")
        raise

    total_available = len(stanford_dogs_dataset)
    
    if total_available == 0:
        raise ValueError("No images found in the dataset. Please check the dataset path and structure.")
    
    # Sample random indices
    print(f"Sampling {total_available} images from dataset of size {total_available}...")
    
    #indices = random.sample(range(total_available), actual_data_size)
    indices = list(range(0, total_available))
    subset = Subset(stanford_dogs_dataset, indices)
    
    # Create a DataLoader to efficiently process the data
    batch_size = 128  # Adjust batch size based on data size
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False  # We already randomized with indices
    )
    
    # Collect the data
    all_images = []
    all_labels = []
    
    print("Processing sampled data...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        all_images.append(images.cpu())  # Move to CPU to save GPU memory
        all_labels.append(labels.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            processed = min((batch_idx + 1) * batch_size, total_available)
            print(f"Processed {processed} / {total_available} samples")
    
    # Concatenate batches
    if len(all_images) == 0:
        raise ValueError("No data was processed successfully")
    
    X = torch.cat(all_images, dim=0)
    y = torch.cat(all_labels, dim=0)
    
    print(f"Mixed dataset created with shape: {X.shape}")
    print(f"Number of unique classes: {len(torch.unique(y))}")
    print(f"Class distribution: min={torch.min(y).item()}, max={torch.max(y).item()}")
    return X, y

def setup_datasets(args):
    """
    Setup federated datasets from Stanford Dogs with both IID and Non-IID support.
    ALWAYS returns 3 values for consistency with main script.
    
    Args:
        args: Arguments object with required attributes
        
    Returns:
        tuple: Always returns (dict_users, server_idx, dataset)
               - For IID: dict_users contains user indices, server_idx contains server indices
               - For Non-IID: dict_users contains (dict_users_train, dict_users_test), 
                              server_idx contains (server_train_idx, server_test_idx)
    """
    distributed_setup_seed(args.seed) 
    dataset = {}                 
    dataset_mixed = {}
    mixed_X, mixed_y = _load_StanfordDogs(args)
    
    # Distribute data among users based on IID/Non-IID setting
    if getattr(args, 'iid', True):
        # IID distribution
        print("Using IID distribution...")
        #dict_users_train, server_id = distribute_stanford_dogs_iid(
        #    mixed_y.size(0), 
        #    args.num_users, 
        #    server_id_size=getattr(args, 'server_id_size', 0), 
        #    seed=args.seed
        #)
        dict_users_train, server_id = distribute_cub_iid_class_balanced(mixed_y, args.num_users, server_id_size=getattr(args, 'server_id_size', 0), seed=args.seed)

        print(f"Successfully distributed {mixed_y.size(0)} samples among {args.num_users} users in IID manner")
        print(f"Samples per user: {[len(dict_users_train[i]) for i in range(min(5, args.num_users))]}")
        
        # For IID, return the original format
        dict_users = dict_users_train
        server_idx = server_id
        
    else:
        # Non-IID distribution using Dirichlet
        print("Using Non-IID distribution (Dirichlet)...")
        alpha = getattr(args, 'alpha', 0.5)  # Dirichlet concentration parameter
        
        dict_users_train, dict_users_test, server_idx = distribute_stanford_dogs_dirichlet(
            mixed_X, mixed_y,
            num_users=args.num_users,
            alpha=alpha,
            server_id_size=getattr(args, 'server_id_size', 0),
            tr_frac=getattr(args, 'tr_frac', 0.8),
            seed=args.seed
        )
        
        print(f"Successfully distributed {mixed_y.size(0)} samples among {args.num_users} users in Non-IID manner (alpha={alpha})")
        
        # For Non-IID, pack the results into tuples to maintain 3-value return
        dict_users = (dict_users_train, dict_users_test)
        #server_idx = (server_train_id, server_test_id)

    dataset_mixed['X'] = mixed_X
    dataset_mixed['y'] = mixed_y
    dataset['mixed'] = dataset_mixed
    
    # Add metadata to help load_data function know which mode we're in
    dataset['iid_mode'] = getattr(args, 'iid', True)
    
    # ALWAYS return exactly 3 values
    return dict_users, server_idx, dataset

def setup_datasets_lazy(args):
    """Setup lazy-loading dataset and distribute indices without loading all images into RAM.

    Instead of materializing all images as a single giant tensor (~11 GB for 16k images),
    this returns a StanfordDogsDataset that reads images from disk on demand.

    Returns:
        tuple: (dict_users, server_idx, lazy_dataset)
            - dict_users: (dict_users_train, dict_users_test) — always a tuple
            - server_idx: list of server indices
            - lazy_dataset: StanfordDogsDataset instance (images loaded on __getitem__)
    """
    distributed_setup_seed(args.seed)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    lazy_dataset = StanfordDogsDataset(args, transform)
    labels = torch.tensor(lazy_dataset.labels)

    if getattr(args, 'iid', True):
        print("Using IID distribution (lazy)...")
        dict_users, server_idx = distribute_cub_iid_class_balanced(
            labels, args.num_users,
            server_id_size=getattr(args, 'server_id_size', 0),
            seed=args.seed,
        )
        # Split each user's indices into train/test (matching non-IID path)
        tr_frac = getattr(args, 'tr_frac', 0.8)
        dict_users_train = {}
        dict_users_test = {}
        for uid, indices in dict_users.items():
            user_labels = [labels[idx].item() for idx in indices]
            tr_idx, te_idx = train_test_split(
                indices, train_size=tr_frac,
                stratify=user_labels, random_state=args.seed,
            )
            dict_users_train[uid] = tr_idx
            dict_users_test[uid] = te_idx

        print(f"Successfully distributed {len(labels)} samples among {args.num_users} users in IID manner (lazy)")
    else:
        print("Using Non-IID distribution (Dirichlet, lazy)...")
        alpha = getattr(args, 'alpha', 0.5)
        dict_users_train, dict_users_test, server_idx = distribute_stanford_dogs_dirichlet(
            None, labels,
            num_users=args.num_users,
            alpha=alpha,
            server_id_size=getattr(args, 'server_id_size', 0),
            tr_frac=getattr(args, 'tr_frac', 0.8),
            seed=args.seed,
            min_per_label=args.min_per_label
        )
        print(f"Successfully distributed {len(labels)} samples among {args.num_users} users in Non-IID manner (lazy, alpha={alpha})")

    dict_users = (dict_users_train, dict_users_test)
    return dict_users, server_idx, lazy_dataset


def load_Stan_data_lazy(lazy_dataset, dict_users, server_idx, m_i, train=True, private=True):
    """Load one client's (or server's) data from a lazy dataset into tensors.

    This replaces load_Stan_data for the lazy-loading path.  Only the requested
    client's images are read from disk and collected into (X, y) tensors, so peak
    RAM holds at most one client's worth of images rather than the full dataset.

    Args:
        lazy_dataset: StanfordDogsDataset instance
        dict_users: (dict_users_train, dict_users_test) tuple
        server_idx: list of server indices
        m_i: client id
        train: True for training split, False for test split
        private: True for client data, False for server data

    Returns:
        tuple: (X_batch, y_batch) tensors
    """
    dict_users_train, dict_users_test = dict_users

    if private:
        if train:
            indices = dict_users_train.get(m_i, [])
        else:
            indices = dict_users_test.get(m_i, [])
    else:
        indices = server_idx if server_idx is not None else []

    if len(indices) == 0:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)

    # Read images from disk one batch at a time via DataLoader
    subset = Subset(lazy_dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=False)
    all_images = []
    all_labels = []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)

    X_batch = torch.cat(all_images, dim=0)
    y_batch = torch.cat(all_labels, dim=0).long()
    return X_batch, y_batch


def apply_augmentation_transforms(image):
    """
    Apply all three types of augmentation transforms to a PIL image simultaneously
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Augmented image with all transformations applied
    """
    # Start with the original image
    augmented_image = image.copy()
    
    # 1. Apply skew transformation
    skew_factor = np.random.uniform(-0.2, 0.2)  # magnitude=0.2
    augmented_image = augmented_image.transform(
        augmented_image.size, 
        Image.AFFINE, 
        (1, skew_factor, 0, 0, 1, 0), 
        resample=Image.BILINEAR
    )
    
    # 2. Apply shear transformation
    shear_factor = np.random.uniform(-10, 10)  # max_shear_left=10, max_shear_right=10
    shear_radians = math.radians(shear_factor)
    augmented_image = augmented_image.transform(
        augmented_image.size, 
        Image.AFFINE,
        (1, math.tan(shear_radians), 0, 0, 1, 0),
        resample=Image.BILINEAR
    )
    
    # 3. Apply random distortion (perspective transformation)
    width, height = augmented_image.size
    distortion_factor = 0.1
    coeffs = [
        1 + np.random.uniform(-distortion_factor, distortion_factor), 
        np.random.uniform(-distortion_factor, distortion_factor),
        0,
        np.random.uniform(-distortion_factor, distortion_factor),
        1 + np.random.uniform(-distortion_factor, distortion_factor),
        0,
        np.random.uniform(-0.0001, 0.0001),
        np.random.uniform(-0.0001, 0.0001)
    ]
    augmented_image = augmented_image.transform(
        augmented_image.size, 
        Image.PERSPECTIVE, 
        coeffs, 
        resample=Image.BILINEAR
    )
    
    # 4. Apply horizontal flip with 50% probability
    if np.random.random() < 0.5:
        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    return augmented_image

def load_Stan_data_prev(args, dataset, server_idx, m_i, dict_users, train=True, private=True):
    """
    Load data for a specific user or server with train/test split.
    This function handles both IID and Non-IID cases automatically.
    
    Args:
        args: Arguments object
        dataset: Dataset dictionary
        server_idx: Server indices (for IID) or tuple (server_train_idx, server_test_idx) for Non-IID
        m_i: User/client ID (device_id)
        dict_users: Dictionary of user indices (for IID) or tuple (dict_users_train, dict_users_test) for Non-IID
        train: Whether to return training or test data
        private: Whether to use user's private data or server data
        
    Returns:
        tuple: (X_batch, y_batch)
    """
    data = dataset['mixed']
    iid_mode = dataset.get('iid_mode', True)
    
    if iid_mode:
        # IID case - need to do train/test split here
        if private == True:
            if m_i not in dict_users:
                print(f"Warning: User {m_i} not found in dict_users")
                return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
            
            user_labels = [data['y'][idx].item() for idx in dict_users[m_i]]
            train_indices, test_indices = train_test_split(
                dict_users[m_i],
                train_size=getattr(args, 'tr_frac', 0.8),
                #stratify=user_labels,  # Key addition!
                    random_state= args.seed
                )
            if train:
                X_batch = data['X'][train_indices]
                y_batch = data['y'][train_indices]
            else:
                X_batch = data['X'][test_indices]
                y_batch = data['y'][test_indices]    #.clone().detach()
        else:
            if server_idx is None:
                print("Warning: No server data available")
                return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
            
            X_batch = data['X'][server_idx].clone().detach()
            y_batch = data['y'][server_idx].clone().detach()
        
    else:
        # Non-IID case - train/test split already done
        dict_users_train, dict_users_test = dict_users
        #server_train_idx, server_test_idx = server_idx
                
        if private == True:
            if train:
                if m_i not in dict_users_train or not dict_users_train[m_i]:
                    print(f"Warning: User {m_i} has no training data")
                    return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
                indices = dict_users_train[m_i]
            else:
                if m_i not in dict_users_test or not dict_users_test[m_i]:
                    print(f"Warning: User {m_i} has no test data")
                    return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
                indices = dict_users_test[m_i]
        else:
            #if train:
            #if not server_train_idx:
                #print("Warning: No server training data available")
                #return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
            #indices = server_train_idx
            #else:
                #if not server_test_idx:
                    #print("Warning: No server test data available")
                    #return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
                #indices = server_test_idx
            if server_idx is None:
                print("Warning: No server training data available")
            indices = server_idx
        
        X_batch = data['X'][indices].clone().detach()
        y_batch = data['y'][indices].clone().detach()

    return X_batch, y_batch

def load_Stan_data(args, dataset, server_idx, m_i, dict_users, train=True, private=True, transform=False):
    """
    Load data for a specific user or server with optional random augmentation.
    
    Args:
        args: Arguments object
        dataset: Dataset dictionary
        server_idx: Server indices [cite: 558]
        m_i: User/client ID [cite: 558]
        dict_users: User indices dictionary [cite: 558]
        train: Whether to return training or test data [cite: 559]
        private: Whether to use user's private data or server data [cite: 559]
        transform (bool): If True, apply random augmentation from augmentation.txt [cite: 575]
        
    Returns:
        tuple: (X_batch, y_batch)
    """
    data = dataset['mixed']
    iid_mode = dataset.get('iid_mode', True)
    
    # 1. Selection of indices
    if iid_mode:
        if private:
            # User private data logic [cite: 560]
            if m_i not in dict_users:
                return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
            
            user_labels = [data['y'][idx].item() for idx in dict_users[m_i]]
            train_indices, test_indices = train_test_split(
                dict_users[m_i],
                train_size=getattr(args, 'tr_frac', 0.8),
                #stratify=user_labels,
                random_state=args.seed
            )
            indices = train_indices if train else test_indices 
        else:
            # Server allocated data logic [cite: 563]
            if server_idx is None:
                return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
            indices = server_idx
    else:
        # Non-IID logic [cite: 564]
        dict_users_train, dict_users_test = dict_users
        if private:
            if train:
                if m_i not in dict_users_train or not dict_users_train[m_i]:
                    print(f"Warning: User {m_i} has no training data")
                    return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
                indices = dict_users_train[m_i]
            else:
                if m_i not in dict_users_test or not dict_users_test[m_i]:
                    print(f"Warning: User {m_i} has no test data")
                    return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)
                indices = dict_users_test[m_i]
        else:
            indices = server_idx if server_idx is not None else []

    if not indices:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)

    # 2. Extract Batch
    X_batch = data['X'][indices].clone().detach()
    y_batch = data['y'][indices].clone().detach() 

    # 3. Apply Augmentation if 'transform' is True
    # This applies to both private and server_idx data if the flag is raised
    if transform and len(X_batch) > 0:
        augmented_list = []
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        # Normalization parameters from TesNet/Preprocess [cite: 542]
        normalize = transforms.Normalize(mean=mean, std=std) 
        
        for img_tensor in X_batch:
            # apply_augmentation_transforms performs skew, shear, perspective, and flip [cite: 575-579]
            pil_img = to_pil(img_tensor)
            aug_pil_img = apply_augmentation_transforms(pil_img) 
            # Re-normalize after transformation to maintain data distribution [cite: 543]
            aug_tensor = normalize(to_tensor(aug_pil_img))
            augmented_list.append(aug_tensor)
            
        X_batch = torch.stack(augmented_list)

    return X_batch, y_batch


class StanfordDogsDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Stanford Dogs with support for both original and cropped images
    """
    def __init__(self, args, transform=None):
        self.transform = transform
        self.use_bbox = args.use_bbox
        
        # Choose image directory based on use_bbox flag
        if args.dataset == 'Stanford_dog':
            if self.use_bbox:
                # Use pre-cropped images
                self.images_dir = '/root/Cropped_Images'
                print(f"Using pre-cropped images from: {self.images_dir}")
            else:
                # Use original images
                self.images_dir = '/home/jilee/jilee/data/Images'
                print(f"Using original images from: {self.images_dir}")
        elif args.dataset == 'Stanford_cars':
            if self.use_bbox:
                # Use pre-cropped images
                self.images_dir = '/root/stanford_cars_cropped/integrated'
                print(f"Using pre-cropped images from: {self.images_dir}")
            else:
                # Use original images
                self.images_dir = '/data2/data/StanfordCars/integrated'
                print(f"Using original images from: {self.images_dir}")
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found at: {self.images_dir}")
        
        print(f"Loading Stanford Dogs dataset from: {self.images_dir}")
        
        # Get all breed folders (like n02085936-Maltese_dog)
        breed_folders = [f for f in os.listdir(self.images_dir) 
                        if os.path.isdir(os.path.join(self.images_dir, f))]
        breed_folders.sort()  # Ensure consistent ordering
        
        print(f"Found {len(breed_folders)} breed folders")
        
        # Process each breed folder
        for class_id, breed_folder in enumerate(breed_folders):
            breed_path = os.path.join(self.images_dir, breed_folder)
            self.class_names.append(breed_folder)
            
            # Get all image files in this breed folder
            image_files = []
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                pattern = os.path.join(breed_path, f"*.{ext}")
                image_files.extend(glob.glob(pattern))
            
            # Add all images for this breed
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(class_id)
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.class_names)} classes")
        
        if len(self.image_paths) == 0:
            raise ValueError("No images found in the dataset directory")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label
