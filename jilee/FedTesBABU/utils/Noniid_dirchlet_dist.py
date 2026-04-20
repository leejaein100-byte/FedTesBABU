import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from util.preprocess import mean, std, preprocess_input_function

def safe_transform(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image

def distribute_data_dirichlet(X, y, num_users, alpha=0.5, server_id_size=0, tr_frac=0.8, seed=42):
    """
    Distribute data indices in a non-IID manner following a Dirichlet distribution.
    Ensures that each user's train and test splits have consistent distribution.
    
    Args:
        X (torch.Tensor): Data tensor
        y (torch.Tensor): Labels tensor
        num_users (int): Number of users to distribute data to
        alpha (float): Concentration parameter of the Dirichlet distribution
                     Lower alpha -> more heterogeneous distribution
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
    
    # Dictionary to store user indices (before train/test split)
    dict_users_all = {i: [] for i in range(num_users)}
    
    # Server indices (before train/test split)
    server_indices_all = []
    
    # Create index list for each class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label.item()].append(idx)
    
    # If server needs data, allocate it first
    if server_id_size > 0:
        # Take a balanced set of samples for the server
        samples_per_class = max(1, server_id_size // n_classes)
        for class_idx, indices in class_indices.items():
            if len(indices) > samples_per_class:
                # Randomly select indices for the server
                selected_indices = np.random.choice(indices, samples_per_class, replace=False)
                server_indices_all.extend(selected_indices)
                
                # Remove selected indices from the class pool
                class_indices[class_idx] = list(set(indices) - set(selected_indices))
    
    # For each class, distribute samples according to Dirichlet distribution
    for class_idx, indices in class_indices.items():
        if not indices:  # Skip if no samples left
            continue
            
        # Draw samples from Dirichlet distribution
        # Lower alpha -> more concentrated distribution (more non-IID)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        
        # Calculate number of samples per user for this class
        # Ensure the sum equals the number of samples
        proportions = np.array([p*(len(indices)-1e-3) for p in proportions])
        proportions = proportions.astype(int)
        
        # Adjust for rounding errors
        diff = len(indices) - sum(proportions)
        proportions[0] += diff
        
        # Distribute samples to users
        index_start = 0
        for user_idx, prop in enumerate(proportions):
            if prop > 0:
                dict_users_all[user_idx].extend(indices[index_start:index_start+prop])
                index_start += prop
    
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
        for class_label, class_indices in user_class_indices.items():
            n_train = int(len(class_indices) * tr_frac)
            
            # Shuffle indices
            np.random.shuffle(class_indices)
            
            # Split
            train_indices = class_indices[:n_train]
            test_indices = class_indices[n_train:]
            
            # Append to user's train and test sets
            dict_users_train[user_idx].extend(train_indices)
            dict_users_test[user_idx].extend(test_indices)
    
    # Split server data into train and test
    server_train_indices = []
    server_test_indices = []
    
    if server_id_size > 0:
        # Group server data by class
        server_class_indices = defaultdict(list)
        for idx in server_indices_all:
            label = y[idx].item()
            server_class_indices[label].append(idx)
        
        # For each class, split into train and test
        for class_label, indices in server_class_indices.items():
            n_train = int(len(indices) * tr_frac)
            
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Split
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            # Append to server's train and test sets
            server_train_indices.extend(train_indices)
            server_test_indices.extend(test_indices)
    
    # Print statistics
    print("\nTrain-Test Distribution Statistics:")
    print(f"Server: {len(server_train_indices)} train, {len(server_test_indices)} test samples")
    
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
        
        print(f"User {user_idx}: {len(train_indices)} train, {len(test_indices)} test samples")
        
        # Show top 3 classes for train and test to verify consistency
        train_top = sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        test_top = sorted(test_class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"  Train top classes: {[f'Class {c}: {n}' for c, n in train_top]}")
        print(f"  Test top classes:  {[f'Class {c}: {n}' for c, n in test_top]}")
    
    return dict_users_train, dict_users_test, server_train_indices, server_test_indices

def load_cropped_CUB():
    rootpath = '/home/swh/Symbolic/Temporary/jilee/TesNet/CUB_200_2011/'
    imgspath = '/home/swh/Symbolic/Temporary/jilee/TesNet/CUB_200_2011/images/'
    all_images = []
    folders = pd.read_table(rootpath + 'images.txt', delimiter=' ', names=['id', 'folder'])
    #folders = folders.to_numpy()
    for _, row in folders.iterrows():
        folder_path = os.path.join(imgspath, row['folder'])
        #image_paths = [os.path.join(imgspath, row['folder'])]
        image = Image.open(folder_path).convert('RGB')
        all_images.append(image)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize])
    all_images = [transform(safe_transform(image)) for image in all_images]
    all_images= torch.stack(all_images, dim=0)
    labels = []
    with open(rootpath + 'images.txt', 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            id_number, path_name = parts[0], parts[1]
            full_name= path_name.split('/')
            file_name = full_name[0]
            file_parts = file_name.split('.')
            full_name_idx = file_parts[0]
            #print(int(full_name_idx))
            labels.append(int(full_name_idx)-1)     #001, 002 등이 여기서 뽑힐듯
        labels = torch.tensor(labels)

    return all_images, labels

def load_cropped_CUB_with_dirichlet(alpha=0.5, num_users=10, server_id_size=0, tr_frac=0.8, seed=42):
    """
    Load CUB dataset and distribute it according to Dirichlet distribution
    
    Args:
        alpha (float): Concentration parameter for Dirichlet distribution
        num_users (int): Number of users to distribute data to
        server_id_size (int): Number of samples for the server
        tr_frac (float): Fraction of data to use for training (per user)
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with keys 'train' and 'test', each containing:
              dict mapping user IDs to tuple of (X, y) for their data
              also includes 'server' if server_id_size > 0
    """
    # Load CUB dataset
    X, y = load_cropped_CUB()
    
    # Distribute data
    dict_users_train, dict_users_test, server_train_idx, server_test_idx = distribute_data_dirichlet(
        X, y, num_users, alpha, server_id_size, tr_frac, seed
    )
    
    # Create user datasets
    user_datasets = {'train': {}, 'test': {}}
    
    # Create training datasets for each user
    for user_idx in range(num_users):
        train_indices = dict_users_train[user_idx]
        test_indices = dict_users_test[user_idx]
        
        if train_indices:
            user_datasets['train'][user_idx] = (X[train_indices], y[train_indices])
            user_datasets['test'][user_idx] = (X[test_indices], y[test_indices])
    
    # Create server dataset if needed
    if server_id_size > 0:
        user_datasets['train']['server'] = (X[server_train_idx], y[server_train_idx])
        user_datasets['test']['server'] = (X[server_test_idx], y[server_test_idx])
    
    return user_datasets



