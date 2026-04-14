import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from util.receptive_field import compute_rf_prototype
from util.helpers import makedir, find_high_activation_crop

def push_prototypes_top3(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    """
    Modified push_prototypes function to store top 3 closest patches for each prototype.
    """
    prototype_network_parallel.eval()
    log('\tpush (top 3 version)')

    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.num_prototypes
    
    # Initialize with 3 slots per prototype for storing top 3 minimum distances
    # saves the closest distance seen so far (now with 3 per prototype)
    #global_min_proto_dist = np.full([n_prototypes, 3], np.inf)
    global_min_proto_dist = np.full([n_prototypes, 5], np.inf)
    
    # saves the patch representation that gives the current smallest distance (now with 3 per prototype)
    global_min_fmap_patches = np.zeros(
        [n_prototypes, 
         5,  # 3 patches per prototype
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes columns:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    
    Now with an additional dimension for the top 3 selections
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5, 6],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 3, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 3, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.num_classes
    
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        # Use the top3 version of the update function
        update_prototypes_on_batch_top3(search_batch_input,
                                     start_index_of_search_batch,
                                     prototype_network_parallel,
                                     global_min_proto_dist,
                                     global_min_fmap_patches,
                                     proto_rf_boxes,
                                     proto_bound_boxes,
                                     save_prototype_class_identity,
                                     class_specific=class_specific,
                                     search_y=search_y,
                                     num_classes=num_classes,
                                     preprocess_input_function=preprocess_input_function,
                                     prototype_layer_stride=prototype_layer_stride,
                                     dir_for_saving_prototypes=proto_epoch_dir,
                                     prototype_img_filename_prefix=prototype_img_filename_prefix,
                                     prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                     prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    # Save the top 3 prototype boxes and receptive fields
    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        # Save all three instances with rank indicators
        for rank in range(3):
            # Extract each rank of data for saving
            rf_boxes_rank = proto_rf_boxes[:, rank, :]
            bound_boxes_rank = proto_bound_boxes[:, rank, :]
            
            # Save with rank indicator in filename
            np.save(os.path.join(proto_epoch_dir, 
                                f"{proto_bound_boxes_filename_prefix}-receptive_field_rank{rank}_{epoch_number}.npy"),
                    rf_boxes_rank)
            np.save(os.path.join(proto_epoch_dir, 
                                f"{proto_bound_boxes_filename_prefix}_rank{rank}_{epoch_number}.npy"),
                    bound_boxes_rank)
        
        # Also save the complete arrays
        np.save(os.path.join(proto_epoch_dir, 
                            f"{proto_bound_boxes_filename_prefix}-receptive_field_all_{epoch_number}.npy"),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, 
                            f"{proto_bound_boxes_filename_prefix}_all_{epoch_number}.npy"),
                proto_bound_boxes)

    log('\tExecuting push (top 3 version)...')
    
    # You can choose to update the prototype vectors with the best patches (rank 0)
    # If you want to implement this, uncomment and modify the line below:
    # prototype_update = np.reshape(global_min_fmap_patches[:, 0, :, :, :], tuple(prototype_shape))
    # prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    
    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))
    

def update_prototypes_on_batch_top3(search_batch_input,
                              start_index_of_search_batch,
                              prototype_network_parallel,
                              global_min_proto_dist,  # now shape [n_prototypes, 3]
                              global_min_fmap_patches,  # now shape [n_prototypes, 3, channels, h, w]
                              proto_rf_boxes,  # now shape [n_prototypes, 3, 5]
                              proto_bound_boxes,  # now shape [n_prototypes, 3, 5]
                              save_prototype_class_identity,
                              class_specific=True,
                              search_y=None,
                              num_classes=None,
                              preprocess_input_function=None,
                              prototype_layer_stride=1,
                              dir_for_saving_prototypes=None,
                              prototype_img_filename_prefix=None,
                              prototype_self_act_filename_prefix=None,
                              prototype_activation_function_in_numpy=None):
    """
    Similar to update_prototypes_on_batch but finds top 3 closest patches for each prototype.
    """
    prototype_per_class=prototype_network_parallel.prototype_per_class
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # Forward pass
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())  # [batchsize,128,14,14]
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())  # [batchsize,2000,14,14]

    del protoL_input_torch, proto_dist_torch

    # Create mapping from class to image indices
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)
        prototype_per_class = prototype_network_parallel.prototype_per_class

    # Get prototype shape
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]  
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]

    # Process each prototype
    for j in range(n_prototypes):
        if class_specific:
            target_class = j
            if target_class not in class_to_img_index_dict or len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]
        else:
            # Use all images
            proto_dist_j = proto_dist_[:, j, :, :]  # [batchsize,14,14]

        # Flatten for finding top k
        flat_proto_dist_j = proto_dist_j.flatten()
        if flat_proto_dist_j.size == 0:
            continue

        k = global_min_proto_dist.shape[1]
        topk_indices = np.argpartition(flat_proto_dist_j, k - 1)[:k]
        topk_indices = topk_indices[np.argsort(flat_proto_dist_j[topk_indices])]
        topk_values = flat_proto_dist_j[topk_indices]
        topk_coords = [list(np.unravel_index(idx, proto_dist_j.shape)) for idx in topk_indices]

        batch_candidates = []
        for coord, dist_value in zip(topk_coords, topk_values):
            batch_argmin_proto_dist_j = list(coord)
            if class_specific:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
            # Extract patch coordinates
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            # Extract the feature map patch
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                :,
                                                fmap_height_start_index:fmap_height_end_index,
                                                fmap_width_start_index:fmap_width_end_index]

            # Get receptive field info
            protoL_rf_info = prototype_network_parallel.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

            # Get bounding box info
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_act_img_j = -proto_dist_img_j
            original_img_j = search_batch_input[rf_prototype_j[0]].numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                        interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                    proto_bound_j[2]:proto_bound_j[3], :]
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                rf_prototype_j[3]:rf_prototype_j[4], :]
            candidate_data = (
                dist_value,
                batch_min_fmap_patch_j,
                # --- RF box data ---
                [rf_prototype_j[0] + start_index_of_search_batch,
                rf_prototype_j[1], rf_prototype_j[2],
                rf_prototype_j[3], rf_prototype_j[4],
                search_y[rf_prototype_j[0]].item()],
                # --- Bound box data ---
                [rf_prototype_j[0] + start_index_of_search_batch,
                proto_bound_j[0], proto_bound_j[1],
                proto_bound_j[2], proto_bound_j[3],
                search_y[rf_prototype_j[0]].item()], original_img_j,
            proto_img_j,
            rf_img_j,
            upsampled_act_img_j,
            proto_act_img_j
            )
            batch_candidates.append(candidate_data)

        global_candidates = []
        for rank in range(k):
            if proto_rf_boxes.shape[2] == 6:
                rf_box_data = proto_rf_boxes[j, rank, :]
                bound_box_data = proto_bound_boxes[j, rank, :]
            else:
                # Add placeholder for class identity if it wasn't saved
                rf_box_data = np.append(proto_rf_boxes[j, rank, :], -1)
                bound_box_data = np.append(proto_bound_boxes[j, rank, :], -1)

            candidate_data = (
                global_min_proto_dist[j, rank],
                global_min_fmap_patches[j, rank],
                rf_box_data,
                bound_box_data, None, None, None, None, None
            )
            global_candidates.append(candidate_data)

        # 3. Combine, sort, and find the new top 3
        all_candidates = batch_candidates + global_candidates
        all_candidates.sort(key=lambda x: x[0])  # Sort by dist_value (ascending)
        new_winners = all_candidates[:k]

        # 4. Update global arrays with the new top 3 winners
# 4. Update global arrays and SAVE IMAGES for new winners
        # Unpack the tuple: dist, fmap_patch, rf_box, bound_box, ...images...
        for rank, (dist, fmap_patch, rf_box, bound_box, 
                   cand_orig_img, cand_proto_img, cand_rf_img, cand_upsampled_act, cand_proto_act) in enumerate(new_winners):
            
            global_min_proto_dist[j, rank] = dist
            global_min_fmap_patches[j, rank] = fmap_patch
            
            if save_prototype_class_identity:
                proto_rf_boxes[j, rank, :] = rf_box
                proto_bound_boxes[j, rank, :] = bound_box
            else:
                proto_rf_boxes[j, rank, :] = rf_box[:5]
                proto_bound_boxes[j, rank, :] = bound_box[:5]
            
            # FIX: Only save images if we have image data (meaning it's a NEW batch winner)
            if cand_orig_img is not None and dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    np.save(os.path.join(dir_for_saving_prototypes, prototype_self_act_filename_prefix + f"_{j}_rank{rank}.npy"), cand_proto_act)

                if prototype_img_filename_prefix is not None:
                    # Save Original
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + f"-original_{j}_rank{rank}.png"),
                               cand_orig_img, vmin=0.0, vmax=1.0)
                    
                    # Overlay
                    rescaled_act_img = cand_upsampled_act - np.amin(cand_upsampled_act)
                    rescaled_act_img = rescaled_act_img / np.amax(rescaled_act_img)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img = 0.5 * cand_orig_img + 0.3 * heatmap
                    
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + f"-original_with_self_act_{j}_rank{rank}.png"),
                               overlayed_original_img, vmin=0.0, vmax=1.0)
                    
                    # Save Receptive Field
                    if cand_rf_img.shape[0] != cand_orig_img.shape[0] or cand_rf_img.shape[1] != cand_orig_img.shape[1]:
                        plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + f"-receptive_field_{j}_rank{rank}.png"),
                                   cand_rf_img, vmin=0.0, vmax=1.0)
                        
                        overlayed_rf_img = overlayed_original_img[int(rf_box[1]):int(rf_box[2]), int(rf_box[3]):int(rf_box[4])]
                        plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + f"-receptive_field_with_self_act_{j}_rank{rank}.png"),
                                   overlayed_rf_img, vmin=0.0, vmax=1.0)
                        
                    # Save Prototype Crop
                    plt.imsave(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + f"_{j}_rank{rank}.png"),
                               cand_proto_img, vmin=0.0, vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict
