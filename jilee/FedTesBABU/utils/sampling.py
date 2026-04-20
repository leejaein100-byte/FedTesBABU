import numpy as np
import pdb
from torchvision import datasets, transforms
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset['y'])/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset['y']))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, num_data, train = True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if train == True:
       num_shards, num_imgs = 200, 250
       server_idx = list(range(num_shards*num_imgs, 60000))
    else:
       num_shards, num_imgs = 200, 40
       server_idx = list(range(num_shards*num_imgs, 10000))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset['y'].numpy()[:num_shards*num_imgs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            add_idx = np.array(list(set(idxs[rand*num_imgs:(rand+1)*num_imgs]) ))
            dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0)

    cnts_dict = {}
    with open("mnist_%d_u%d.txt"%(num_data, num_users), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
    
    
    #server_idx = list(range(num_shards*num_imgs, 60000))
    return dict_users, server_idx, cnts_dict

def cifar_iid(y, num_users, server_id_size): #,train=True):  #num_data=50000
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    all_idxs = [i for i in range(y.size(0))]
  
    #if train == True:
      #total_data = 50000
    if server_id_size > 0 :    #num_data < 50000:
      if server_id_size > y.size(0):
        print("invalid server data allocation")
      else:
        server_idx = np.random.choice(all_idxs, server_id_size, replace=False)
        all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)

    for i in range(num_users):
      dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
      all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #if train == True:
    #for i in range(num_users):
      #dict_users_val[i] = set(np.random.choice(dict_users[i], val_num_items, replace=False))
      #dict_users[i] = list(set(dict_users[i]) - set(dict_users_val[i]))   

    return dict_users, server_idx 
    
def cifar_noniid(y, num_users,num_data, num_classes, method="step"): #num_data=50000, train=True,얘는 그냥 있는 데이터만 순수하게
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """

    labels = np.array(y)   #dataset.targets
    _lst_sample = 20
    total_data=np.shape(labels)[0]
    #if train == True:
      #total_data = 50000    #49968   #50000
    num_items = int(total_data/num_users)
    val_num_items= int(0.1* num_items)
      #total_data = 10000 #9984  #10000
    dict_users_val={}

    if method=="step":
      
      num_shards = num_users*2
      num_imgs = total_data// num_shards
      idx_shard = [i for i in range(num_shards)]
      

      idxs = np.arange(num_shards*num_imgs)
      # sort labels
      idxs_labels = np.vstack((idxs, labels[:total_data]))
      idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  #argsort는 오름차순 정렬
      idxs = idxs_labels[0,:]  #0인 index들, 1인 index들 ...
      
      least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=np.int32)
      for i in range(num_classes):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #if train:
      server_idx = np.random.choice(list(set(range(total_data))-least_idx_set), total_data-num_data, replace=False)

      # divide and assign
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
      for i in range(num_users):
          rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
          idx_shard = list(set(idx_shard) - rand_set)
          for rand in rand_set:
              idx_i = list( set(range(rand*num_imgs, (rand+1)*num_imgs))   )
              #if train:
              add_idx = list(set(idxs[idx_i]) - set(server_idx)-set(least_idx_set) )   #이게 들어가야 맞을거 같은데... 내일 논문으로 얘네 실험환경 조금만 더 자세히 보자...             
              #else:
                #add_idx = list(set(idxs[idx_i]) -set(least_idx_set) )
              dict_users[i] = np.concatenate((dict_users[i], add_idx), axis=0) 
          dict_users[i] = np.concatenate((dict_users[i], least_idx[i]), axis=0) #아니 이러면 겹치는거 생기는거 아니야?? 뭐 10개니까 별 상관 없을거 같긴 한데...
          
      #if train == True:
      #for i in range(num_users):
        #dict_users_val[i] = set(np.random.choice(dict_users[i], val_num_items, replace=False))
        #dict_users[i] = list(set(dict_users[i]) - set(dict_users_val[i]))   
      return dict_users, server_idx
      #else:
        #return dict_users
  
    elif method == "dir":
      min_size = 0
      K = num_classes
      y_train = labels
      
      _lst_sample = 2

      least_idx = np.zeros((num_users, K, _lst_sample), dtype=np.int)
      for i in range(K):  # 10은 Cifar class 개수
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
      least_idx = np.reshape(least_idx, (num_users, -1))
      
      least_idx_set = set(np.reshape(least_idx, (-1)))
      #least_idx_set = set([])
      server_idx = np.random.choice(list(set(range(total_data))-least_idx_set), total_data-num_data, replace=False)
      local_idx = np.array([i for i in range(total_data) if i not in server_idx and i not in least_idx_set])
      
      N = y_train.shape[0]
      dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

      while min_size < 10:
          idx_batch = [[] for _ in range(num_users)]
          # for each class in the dataset
          for k in range(K):
              idx_k = np.where(y_train == k)[0]
              idx_k = [id for id in idx_k if id in local_idx]
              np.random.shuffle(idx_k)
              proportions = np.random.dirichlet(np.repeat(0.1, num_users))
              proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])  #d이렇게 하면 이 '<' 부등호가 조건 역할을 하나?
              proportions = proportions/proportions.sum()
              proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]  #얘가 한 [] 안에 끊어야 할 index를 알려준다 [4 11 18 27] 1 user:4번까지& 2번 user: 11번까지
              idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
              min_size = min([len(idx_j) for idx_j in idx_batch])

      for j in range(num_users):
          np.random.shuffle(idx_batch[j])
          dict_users[j] = idx_batch[j]  
          dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
      for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(10)] )
        cnts_dict[i] = cnts
        f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
   
      
def TIM_iid(data_size, num_users, server_id_size): #,train=True):  #num_data=50000   총 data수에서 data_size만큼 선택하게 하는 모듈이 필요
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    all_idxs = [i for i in range(data_size)]
  
    #if train == True:
      #total_data = 50000
    if server_id_size > 0 :                          #num_data < 50000:
      server_idx = np.random.choice(all_idxs, server_id_size, replace=False)
      all_idxs = list(set(all_idxs) - set(server_idx))
    num_items = int(len(all_idxs)/num_users)

    #else:
      #total_data = data_size
      #if num_data < 10000:                          #num_data < 50000:
        #server_idx = np.random.choice(all_idxs, data_size-num_data, replace=False)
        #all_idxs = list(set(all_idxs) - set(server_idx))
      #num_items = int(len(all_idxs)/num_users)
  
    #for p_i in range(p):
      #all_idxs_tmp=all_idxs
      #for i in range(m_per_cluster):
    for i in range(num_users):
      dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
      all_idxs = list(set(all_idxs) - set(dict_users[i]))
    #if train == True:
    #for i in range(num_users):
      #dict_users_val[i] = set(np.random.choice(dict_users[i], val_num_items, replace=False))
      #dict_users[i] = list(set(dict_users[i]) - set(dict_users_val[i]))   

    return dict_users, server_idx 