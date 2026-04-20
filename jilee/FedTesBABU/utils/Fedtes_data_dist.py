import numpy as np
import random
from utils.sampling import *

from torchvision import datasets, transforms
import torchvision.datasets 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, ConcatDataset
#from utils.options import args_parser
from sklearn.model_selection import train_test_split
from utils.utils import *
import os
import pandas as pd
from PIL import Image
import Augmentor
import settings_CUB
from util.preprocess import mean, std, preprocess_input_function

CIFAR10_MIXED_DATA_SIZE = 60000
TIM_MIXED_DATA_SIZE = 110000
#np.random.seed(0)
SEED = 0

def _load_ImageNet1k(data_size=200000, root_path='/data/dataset/imagenet'):
    """
    Load a mixed dataset of ImageNet1k with both training and validation samples.
    
    Args:
        data_size (int): Total number of samples to include in the mixed dataset
        root_path (str): Path to the ImageNet directory
        
    Returns:
        tuple: (X, y) tensors containing images and labels
    """
    # Define transforms - similar to ImageNet standard preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load both training and validation datasets
    print("Loading ImageNet training and validation datasets...")
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_path, 'train'),
        transform=transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_path, 'val'),
        transform=transform
    )
    
    # Concatenate the datasets
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # Sample random indices
    print(f"Sampling {data_size} images from combined dataset of size {len(combined_dataset)}...")
    indices = random.sample(range(len(combined_dataset)), min(data_size, len(combined_dataset)))
    subset = Subset(combined_dataset, indices)
    
    # Create a DataLoader to efficiently process the data
    dataloader = DataLoader(
        subset,
        batch_size=128,  # Adjust based on your available memory
        num_workers=4,
        pin_memory=True
    )
    
    # Collect the data
    all_images = []
    all_labels = []
    
    print("Processing sampled data...")
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    
    # Concatenate batches
    X = torch.cat(all_images, dim=0)
    y = torch.cat(all_labels, dim=0)
    
    print(f"Mixed dataset created with shape: {X.shape}")
    return X, y
    
class CustomDataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #image_path = self.image_paths[idx]
        #image = Image.open(image_path).convert('RGB')
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        return X, y
        
def setup_datasets(args):
    dataset = {}                 
    dataset_mixed = {}
    if args.dataset == 'cifar10':
        if args.num_data > CIFAR10_MIXED_DATA_SIZE:
            print("invalid server data allocation")
        else:
            mixed_X, mixed_y= _load_CIFAR10(data_size= args.num_data)   #이녀석 train은 딱히 의미 없음
        #test_X, test_y,test_dataset = _load_CIFAR10(train=False)
            if args.iid:
                dict_users_train, server_id = cifar_iid(mixed_y, args.num_users, server_id_size = args.server_id_size)  #, train=True)   
            else:
                dict_users_train, server_id = cifar_noniid(mixed_y, args.num_users, num_data=args.num_data, method=args.split_method) #,train=True) 

    elif args.dataset == 'cifar100':
        if args.num_data > CIFAR10_MIXED_DATA_SIZE:
            print("invalid server data allocation")
        else:
            mixed_X, mixed_y= _load_CIFAR100(data_size= args.num_data)   #이녀석 train은 딱히 의미 없
            if args.iid:
                #dict_users_train, dict_users_val, server_id, all_idxs = cifar_iid(CIFAR10_TRAINSET_DATA_SIZE,p, m, num_data = args.num_data, train=True)  #Cifar에서만 non-iid를 적용한 논문
                #dict_users_test, all_idxs = cifar_iid(CIFAR10_TESTSET_DATA_SIZE, p, m, num_data = CIFAR10_TESTSET_DATA_SIZE, train=False)  #Cifar에서만 non-iid를 적용한 논문, server_id
                dict_users_train, server_id = cifar_iid(mixed_y, args.num_users, server_id_size = args.server_id_size)  #, train=True)
                #cnts_dict = None  dict_users_train은 전체 dataset중에서 server에 public dataset으로 할당될 것은 제외. 
            else:
                #dict_users_train, dict_users_val, server_id = cifar_noniid(train_y[:CIFAR10_TRAINSET_DATA_SIZE], args.num_users, num_data=args.num_data, method=args.split_method,train=True)  #, cnts_dict
                #dict_users_test= cifar_noniid(test_y, args.num_users, num_data=args.num_data, method=args.split_method,train=False) #, server_id, cnts_dict
                dict_users_train, server_id = cifar_noniid(mixed_y, args.num_users, num_data=args.num_data, method=args.split_method) #,train=True) 
        
    elif args.dataset == 'CUB':  #data size도 같고 random shuffling으로 해결
        mixed_X, mixed_y= load_cropped_CUB()
        if args.iid:
            #partition_size = (11788-args.server_id_size) // args.num_users
            if args.server_id_size > 0 :
                dict_users_train = {}
                all_idxs = [i for i in range(11788)]#num_data < 50000:
                server_id = np.random.choice(all_idxs, args.server_id_size, replace=False)
            #all_idxs = list(set(all_idxs) - set(server_id))
            num_items = int(len(all_idxs)/args.num_users)
            for i in range(args.num_users):
                dict_users_train[i] = np.random.choice(all_idxs, num_items, replace=False)
                all_idxs = list(set(all_idxs) - set(dict_users_train[i]))
        else:
            server_id = np.random.choice(list(set(range(11788))), args.server_id_size, replace=False)    #11788-args.num_data,
            local_idx = np.array([i for i in range(11788)])  #if i not in server_id
            N = mixed_y.shape[0]
            K = args.num_classes
            dict_users_train = {i: np.array([], dtype='int64') for i in range(args.num_users)}
            alpha = getattr(args, 'alpha', 0.5)
            print(f"Using Non-IID Dirichlet distribution for CUB with alpha={alpha}")
            #while min_size < 7:
            idx_batch = [[] for _ in range(args.num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(mixed_y == k)[0]
                idx_k = [id for id in idx_k if id in local_idx]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, args.num_users))
                proportions = np.array([p*(len(idx_j)<N/args.num_users) for p,idx_j in zip(proportions,idx_batch)])  #d이렇게 하면 이 '<' 부등호가 조건 역할을 하나?
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]  #얘가 한 [] 안에 끊어야 할 index를 알려준다 [4 11 18 27] 1 user:4번까지& 2번 user: 11번까지
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                #min_size = min([len(idx_j) for idx_j in idx_batch])  ##idx_batch에 원하는 index로 다 되어 있음

            for j in range(args.num_users):
                np.random.shuffle(idx_batch[j])
                dict_users_train[j] = idx_batch[j]  
                #dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          
        
    elif args.dataset == 'tiny_imagenet':
        if args.num_data > TIM_MIXED_DATA_SIZE:
            print("invalid server data allocation")
        else:
            mixed_X, mixed_y= load_cropped_TinyIm(data_size = args.num_data)
            if args.iid:
                dict_users_train, server_id = cifar_iid(mixed_y, args.num_users, server_id_size = args.server_id_size)  #, train=True)
            else:
                dict_users_train, server_id = cifar_noniid(mixed_y, args.num_users, num_data=args.num_data, num_classes = args.num_classes, method='dir') #,train=True)  

    elif args.dataset == 'imagenet-1k':
        mixed_X, mixed_y= _load_ImageNet1k(data_size=200000, root_path='/data/dataset/imagenet')
        if args.iid:
            dict_users_train, server_id = distribute_imagenet_iid(mixed_y.size(0), args.num_users, server_id_size=0, seed=42)  #, train=True)
        else:
            pass

    dataset_mixed['X'] = mixed_X
    dataset_mixed['y'] = mixed_y
    dataset['mixed'] = dataset_mixed
    return dict_users_train, server_id, dataset
    
  #왜 여기서 server_id가 필요?
def distribute_imagenet_iid(n_samples, num_users, server_id_size=0, seed=42):
    """
    Same as before, distributes data indices in IID manner
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create randomly permuted indices
    all_indices = torch.randperm(n_samples).tolist()
    
    # Dictionary to store user indices
    dict_users = {}
    
    # Server indices
    server_indices = []
    
    # If server needs data, allocate it first
    if server_id_size > 0:
        server_indices = all_indices[:server_id_size]
        all_indices = all_indices[server_id_size:]
    
    # Calculate how many items each user gets
    num_items_per_user = len(all_indices) // num_users
    
    # Distribute remaining data to users
    for i in range(num_users):
        start_idx = i * num_items_per_user
        end_idx = (i + 1) * num_items_per_user if i < num_users - 1 else len(all_indices)
        dict_users[i] = all_indices[start_idx:end_idx]
    
    return dict_users, server_indices


def _load_CIFAR10(data_size):    # 1. train과 test를 함께 섞는다 2. tensor화를 한다?
    transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            #torchvision.transforms.Resize(size=(224, 224)),
                            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 
    CIFAR10_dataset =ConcatDataset([
            torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform= transforms),
            torchvision.datasets.CIFAR10(root='./data', download=False, train=False, transform= transforms)
        ])
    #dataset_path = '/home/jilee/FedTesnet/TesNet/Cifar10/'
    #if not os.listdir(dataset_path):
        #makedir(dataset_path) 

    indices = random.sample(range(len(CIFAR10_dataset)), data_size)
    subset = Subset(CIFAR10_dataset, indices)
    X = torch.cat([sample[0].unsqueeze(0) for sample in subset], dim=0)
    y = torch.tensor([sample[1] for sample in subset])

    X = X / 255.0

    return X, y

def _load_CIFAR100(data_size):
    transforms = torchvision.transforms.Compose([
                            #torchvision.transforms.RandomCrop(32, padding=4),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])   #                            torchvision.transforms.RandomHorizontalFlip(),RandomRotation()
    #if train:
        #CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transforms)
    #else:
        #CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms)
    CIFAR100_dataset =ConcatDataset([
            torchvision.datasets.CIFAR100(root='./data', download=True, train=True, transform=transforms),
            torchvision.datasets.CIFAR100(root='./data', download=False, train=False, transform=transforms)
        ])
 
    #dl = DataLoader(CIFAR10_dataset)

    #X = torch.tensor(dl.dataset.data) # (60000,32,32)
    #y = torch.tensor(dl.dataset.targets) #(60000)
    # normalize to have 0 ~ 1 range in each pixel
    indices = random.sample(range(len(CIFAR100_dataset)), data_size)
    subset = Subset(CIFAR100_dataset, indices)
    X = torch.cat([sample[0].unsqueeze(0) for sample in subset], dim=0)
    y = torch.tensor([sample[1] for sample in subset])
    #X=X.permute(0,1,2,3)

    X = X / 255.0

    return X, y

def safe_transform(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image
#/home/cmluser56/FedTesnet/TesNet/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
def load_cropped_CUB():
    rootpath = '/home/jilee/jilee/TesNet/CUB_200_2011/'
    imgspath = '/home/jilee/jilee/TesNet/CUB_200_2011/images/'
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


def load_cropped_TinyIm2():
    TRAINING_IMAGES_DIR = './tiny-imagenet/tiny-imagenet-200/train'
    VAL_IMAGES_DIR = './tiny-imagenet-200/val/'
    TEST_IMAGES_DIR = './tiny-imagenet-200/test/'
    names = []     
    all_images = []
    all_labels = []
    dirs = [TRAINING_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR]
    img_size = settings_CUB.img_size
    for dir in dirs:
        labels = []           
        for type in os.listdir(dir):
            if os.path.isdir(dir + type + '/images/'):
                type_images = os.listdir(dir + type + '/images/')
                # Loop through all the images of a type directory
                #print ("Loading Class ", type)
                for image in type_images:
                    image_file = os.path.join(dir, type + '/images/', image)
                    image_data = Image.open(image_file).convert('RGB')
                    #print ('Loaded Image', image_file, image_data.shape)
                    if (image_data.size == (3,64,64)):
                        normalize = transforms.Normalize(mean=mean,std=std)
                        transform = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize])
                        all_images.append(transform(image_data))
                        labels.append(type)
                        names.append(image)
        if dir == TRAINING_IMAGES_DIR:    
            label_map = {label: idx for idx, label in enumerate(list(set(labels)))}        
    labels.append(label_map[type])        
    #return (images, np.asarray(labels), np.asarray(names))
    return all_images, labels
    
def load_cropped_TinyIm(data_size):
    TRAINING_IMAGES_DIR = './tiny-imagenet/tiny-imagenet-200/train/'
    VAL_IMAGES_DIR = './tiny-imagenet/tiny-imagenet-200/val/'
    wnids_path = '/home/swh/Symbolic/Temporary/jilee/TesNet/tiny-imagenet/tiny-imagenet-200/' 
    path = '/home/swh/Symbolic/Temporary/jilee/TesNet'   
    num_classes=200
    wnids_file = os.path.join(wnids_path, 'wnids' + str(num_classes) + '.txt')
    with open(wnids_file, 'r') as f:
        wnids = [x.strip() for x in f]
    all_images = []
    labels_list = []
    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    # Use words.txt to get names for each class
    words_file = os.path.join(wnids_path, 'words' + str(num_classes) + '.txt')
    with open(words_file, 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]
    for i, wnid in enumerate(wnids):
        #if (i + 1) % 20 == 0:
            #print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
            # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(wnids_path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
            num_images = len(filenames)
            X_train_block = torch.zeros(num_images, 3, 64, 64)
        
        y_train_block = torch.full((num_images,), wnid_to_label[wnid], dtype=torch.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(wnids_path, 'train', wnid, 'images', img_file)
            img = Image.open(img_file).convert('RGB')
            if (img.size == (3,64,64)):
                normalize = transforms.Normalize(mean=mean,std=std)
                transform = transforms.Compose([
                    transforms.ToTensor(), normalize])
                X_train_block[j] = transform(img.transpose(2, 0, 1))
            else:
                pass
        all_images.append(X_train_block)
        labels_list.append((y_train_block))
    with open(os.path.join(wnids_path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
        # Select only validation images in chosen wnids set
            if line.split()[1] in wnids:
                img_file, wnid = line.split('\t')[:2]
                img_files.append(img_file)
                val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = torch.full((num_val,), wnid_to_label[wnid], dtype=torch.int64)
        
        X_val = torch.zeros(num_val, 3, 64, 64)    
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(wnids_path, 'val', 'images', img_file)
            img = Image.open(img_file).convert('RGB')
            if (img.size == (3,64,64)):
                normalize = transforms.Normalize(mean=mean,std=std)
                transform = transforms.Compose([
                    transforms.ToTensor(), normalize])
                X_val[i] = img.transpose(2, 0, 1)
            else:
                pass
    all_images.append(X_val)
    labels_list.append((y_val))
    all_images = torch.cat(all_images, dim=0)
    labels = torch.cat(labels_list, dim=0)
    TinyIm_dataset= MyDataset(all_images, labels)
    indices = random.sample(range(len(labels)), data_size)
    subset = Subset(TinyIm_dataset, indices)
    X = torch.cat([sample[0].unsqueeze(0) for sample in subset], dim=0)
    y = torch.tensor([sample[1] for sample in subset])    
    #return (images, np.asarray(labels), np.asarray(names))
    return X, y
    
    
def load_data(args, dataset, server_idx, m_i, dict_users, train= True, private = True):  #m_i가 device_id로 대체가 되는지는 의문->그냥 1~num_user로 분배한거라 같은 것.  dict_users_val, ,dict_users_test, val= False)
        # this part is very fast since its just rearranging models, val= False
    #cfg = self.config
    data={}
    data=dataset['mixed']
    #assert (m // p) * n == num_data
    #print(len(dict_users[m_i]))
    if private == True:
        train_indices, test_indices = train_test_split(dict_users[m_i],
        train_size=args.tr_frac, random_state=SEED) 
    else:
        train_indices, test_indices = train_test_split(server_idx, train_size=args.tr_frac,
        random_state=SEED) 
    
    if train:
        X_batch = data['X'][train_indices].clone().detach()
        y_batch = data['y'][train_indices].clone().detach()
    else:
        X_batch = data['X'][test_indices].clone().detach()
        y_batch = data['y'][test_indices].clone().detach()
    return X_batch, y_batch


def apply_transformations(images, labels, num_iterations=10):  
    # Create an Augmentor pipeline without specifying the output directory
    p = Augmentor.Pipeline()
    q = Augmentor.Pipeline()
    r = Augmentor.Pipeline()
    
    # Add transformations to the pipeline
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    q.shear(probability=1, max_shear_left=10, max_shear_right=10)
    q.flip_left_right(probability=0.5)
    r.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    r.flip_left_right(probability=0.5)
    
    # Apply the transformations to the images multiple times
    transformed_images = []
    transformed_labels = []

    # Apply transformations to each image
    for image, label in zip(images, labels):
        temp_transformed_images = []
        temp_transformed_labels = []
        
        # Apply transformations multiple times (num_iterations)
        for _ in range(num_iterations):
            # Apply transformation p
            transformed_image_p = p._execute_with_array(image)
            temp_transformed_images.append(transformed_image_p)
            temp_transformed_labels.append(label)
            
            # Apply transformation q
            transformed_image_q = q._execute_with_array(image)
            temp_transformed_images.append(transformed_image_q)
            temp_transformed_labels.append(label)
            
            # Apply transformation r
            transformed_image_r = r._execute_with_array(image)
            temp_transformed_images.append(transformed_image_r)
            temp_transformed_labels.append(label)

        transformed_images.extend(temp_transformed_images)
        transformed_labels.extend(temp_transformed_labels)

    del p, q, r

    return transformed_images, transformed_labels
