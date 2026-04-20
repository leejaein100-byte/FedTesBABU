import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageOps
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import math

def safe_transform(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def augment_images_in_memory(images, num_augmentations=30):
    """
    Apply augmentation to images in memory with all three types applied simultaneously
    
    Args:
        images (list): List of PIL images
        num_augmentations (int): Number of augmented versions per image
    
    Returns:
        list: List of augmented images (original + augmented)
    """
    augmented_images = []
    
    for img in images:
        # Add original image
        augmented_images.append(img)
        
        # Add augmented versions - each version has all three augmentations applied
        for _ in range(num_augmentations):
            # Apply all augmentation types simultaneously
            augmented_img = apply_augmentation_transforms(img)
            augmented_images.append(augmented_img)
    
    return augmented_images

def load_cropped_CUB_random_distribution(num_users=10, server_id_size=0, tr_frac=0.8, seed=42, apply_augmentation=True):
    """
    Load CUB dataset and distribute it randomly among users with train/test split and augmentation
    
    Args:
        num_users (int): Number of users to distribute data to
        server_id_size (int): Number of samples for the server
        tr_frac (float): Fraction of data to use for training (per user)
        seed (int): Random seed for reproducibility
        apply_augmentation (bool): Whether to apply augmentation to training data
        
    Returns:
        dict: Dictionary with user datasets
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load images and labels
    rootpath = '/home/swh/Symbolic/Temporary/jilee/TesNet/CUB_200_2011/'
    total_images_dir = '/home/swh/Symbolic/Temporary/jilee/TesNet/CUB_200_2011/total_images/'
    
    # Read image names and labels
    names = pd.read_table(rootpath + 'images.txt', delimiter=' ', names=['id', 'name'])
    
    all_images = []
    labels = []
    
    print("Loading images...")
    for idx, row in names.iterrows():
        image_path = os.path.join(total_images_dir, row['name'])
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            all_images.append(image)
            
            # Extract label from filename
            folder_name = row['name'].split('/')[0]
            class_id = int(folder_name.split('.')[0]) - 1  # Convert to 0-indexed
            labels.append(class_id)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    
    print(f"Loaded {len(all_images)} images from {len(set(labels))} classes")
    
    # Create indices for random distribution
    total_samples = len(all_images)
    all_indices = list(range(total_samples))
    np.random.shuffle(all_indices)
    
    # Allocate server data first if needed
    server_indices = []
    if server_id_size > 0:
        server_indices = all_indices[:server_id_size]
        all_indices = all_indices[server_id_size:]
    
    # Distribute remaining data among users
    samples_per_user = len(all_indices) // num_users
    user_indices = {}
    
    for user_id in range(num_users):
        start_idx = user_id * samples_per_user
        end_idx = (user_id + 1) * samples_per_user if user_id < num_users - 1 else len(all_indices)
        user_indices[user_id] = all_indices[start_idx:end_idx]
    
    # Create datasets for each user
    user_datasets = {'train': {}, 'test': {}}
    
    print("Creating user datasets with train/test split...")
    for user_id in range(num_users):
        user_idx_list = user_indices[user_id]
        
        # Get user's images and labels
        user_images = [all_images[i] for i in user_idx_list]
        user_labels = [labels[i] for i in user_idx_list]
        
        # Split into train and test
        train_images, test_images, train_labels, test_labels = train_test_split(
            user_images, user_labels, train_size=tr_frac, random_state=seed + user_id
        )
        
        print(f"User {user_id}: {len(train_images)} train, {len(test_images)} test images")
        
        # Apply augmentation to training data if requested
        if apply_augmentation and len(train_images) > 0:
            print(f"Applying augmentation to user {user_id} training data...")
            train_images_augmented = augment_images_in_memory(train_images, num_augmentations=30)
            # Create corresponding labels for augmented images
            train_labels_augmented = []
            for i, label in enumerate(train_labels):
                train_labels_augmented.append(label)  # Original image label
                train_labels_augmented.extend([label] * 30)  # 30 augmented versions
            train_images = train_images_augmented
            train_labels = train_labels_augmented
            print(f"User {user_id}: {len(train_images)} total training images after augmentation")
        
        # Convert to tensors
        if len(train_images) > 0:
            # Apply standard transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            train_tensors = torch.stack([transform(img) for img in train_images])
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            
            test_tensors = torch.stack([transform(img) for img in test_images])
            test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
            
            user_datasets['train'][user_id] = (train_tensors, train_labels_tensor)
            user_datasets['test'][user_id] = (test_tensors, test_labels_tensor)
        
        else:
            print(f"Warning: User {user_id} has no data")
    
    # Create server dataset if needed
    if server_id_size > 0:
        print("Creating server dataset...")
        server_images = [all_images[i] for i in server_indices]
        server_labels = [labels[i] for i in server_indices]
        
        # Split server data into train/test
        server_train_images, server_test_images, server_train_labels, server_test_labels = train_test_split(
            server_images, server_labels, train_size=tr_frac, random_state=seed
        )
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        server_train_tensors = torch.stack([transform(img) for img in server_train_images])
        server_train_labels_tensor = torch.tensor(server_train_labels, dtype=torch.long)
        
        server_test_tensors = torch.stack([transform(img) for img in server_test_images])
        server_test_labels_tensor = torch.tensor(server_test_labels, dtype=torch.long)
        
        user_datasets['train']['server'] = (server_train_tensors, server_train_labels_tensor)
        user_datasets['test']['server'] = (server_test_tensors, server_test_labels_tensor)
        
        print(f"Server: {len(server_train_images)} train, {len(server_test_images)} test images")
    
    return user_datasets

def load_data_random(dataset,m_i, private=True):
    """
    Load data for a specific user from the randomly distributed dataset
    
    Args:
        args: Arguments
        dataset: Dictionary with user datasets
        server_idx: Server data (not used in this random distribution)
        m_i: User ID
        private: Whether to use user's data (True) or server data (False)
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    if private:
        # Use user's data
        if m_i in dataset['train']:
            X_train, y_train = dataset['train'][m_i]
            X_test, y_test = dataset['test'][m_i]
        else:
            # Return empty tensors if user has no data
            X_train = torch.empty(0, 3, 224, 224)
            y_train = torch.empty(0, dtype=torch.long)
            X_test = torch.empty(0, 3, 224, 224)
            y_test = torch.empty(0, dtype=torch.long)
    else:
        # Use server data
        if 'server' in dataset['train']:
            X_train, y_train = dataset['train']['server']
            X_test, y_test = dataset['test']['server']
        else:
            # Return empty tensors if no server data
            X_train = torch.empty(0, 3, 224, 224)
            y_train = torch.empty(0, dtype=torch.long)
            X_test = torch.empty(0, 3, 224, 224)
            y_test = torch.empty(0, dtype=torch.long)
    
    return X_train, y_train, X_test, y_test

def create_federated_datasets_random(args):
    """
    Create federated datasets with random distribution
    
    Args:
        args: Arguments containing parameters
        
    Returns:
        dict: Dictionary with dataset information
    """
    # Load and distribute dataset randomly
    dataset = load_cropped_CUB_random_distribution(
        num_users=args.num_users,
        server_id_size=args.server_id_size,
        tr_frac=args.tr_frac,
        seed=args.seed,
        apply_augmentation=True
    )
    
    return dataset

# Example usage and testing
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.num_users = 5
            self.server_id_size = 100
            self.tr_frac = 0.8
            self.seed = 42
    
    args = Args()
    
    print("Creating federated datasets with random distribution...")
    dataset = create_federated_datasets_random(args)
    
    # Example of loading data for user 0
    print("\nLoading data for user 0...")
    X_train, y_train, X_test, y_test = load_data_random(
        args, dataset, None, 0, private=True
    )
    
    print(f"User 0 data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Check class distribution
    if len(y_train) > 0:
        unique_classes, counts = torch.unique(y_train, return_counts=True)
        print(f"  Training classes: {len(unique_classes)}")
        print(f"  Top 5 most common classes: {unique_classes[:5].tolist()}")
        print(f"  Their counts: {counts[:5].tolist()}")
    
    # Example of loading server data
    if args.server_id_size > 0:
        print("\nLoading server data...")
        X_train_server, y_train_server, X_test_server, y_test_server = load_data_random(
            args, dataset, None, 0, private=False
        )
        
        print(f"Server data shapes:")
        print(f"  X_train: {X_train_server.shape}, y_train: {y_train_server.shape}")
        print(f"  X_test: {X_test_server.shape}, y_test: {y_test_server.shape}")
    
    print("\nDataset creation complete!")