import numpy as np
from sklearn.cluster import DBSCAN
from scipy.special import softmax

from detection import Detection

def get_labels(data_dict):
    if 'gt_labels' in data_dict:
        return data_dict['gt_labels']
    if 'name' in data_dict:
        classes = ['Car', 'Pedestrian', 'Cyclist']
        return np.array([classes.index(name)+1 for name in data_dict['name']])
    raise ValueError()

# Input is list(list(dicts))
# First list is number of frames in the dataset
# Second list is number of outputs per frame
# Each dict is a normal frame output from OpenPCDet
# Output is list(dicts)
# First list is number of frames in the dataset
# Each dict is in the form of a frame output from OpenPCDet
# but the values inside are from the mean of each cluster
def cluster_preds(pred_dicts):
    num_frames = len(pred_dicts)
    num_outputs_per_frame = len(pred_dicts[0])

    # If output is already one dict per frame
    if num_outputs_per_frame == 1:
        return pred_dicts

    new_pred_dicts = []
    for frame_dict_list in pred_dicts:
        name_list = []
        label_list = []
        score_list = []
        score_all_list = []
        box_list = []
        box_var_list = []
        for single_data_dict in frame_dict_list:
            names_one_frame = single_data_dict[0]['name']
            labels_one_frame = get_labels(single_data_dict[0])
            score_one_frame = single_data_dict[0]['score']
            score_all_one_frame = single_data_dict[0]['score_all']
            boxes_one_frame = single_data_dict[0]['boxes_lidar']
            box_vars_one_frame = single_data_dict[0]['pred_vars']
            for i in range(len(labels_one_frame)):
                name_list.append(names_one_frame[i])
                label_list.append(labels_one_frame[i])
                score_list.append(score_one_frame[i])
                score_all_list.append(softmax(score_all_one_frame[i])) # Apply softmax
                box_list.append(boxes_one_frame[i])
                box_var_list.append(box_vars_one_frame[i])
        name_list = np.array(name_list)
        label_list = np.array(label_list)
        score_list = np.array(score_list)
        score_all_list = np.array(score_all_list)
        box_list = np.array(box_list)
        box_var_list = np.array(box_var_list)
        #     print(box_list)

        ctable = Detection.compute_ctable(box_list, box_list, criterion='iou')
        #     print(ctable)

        # IDs >= 0 are valid clusters while -1 means the cluster did not reach min samples
        cluster_ids = DBSCAN(eps=0.5, min_samples=num_outputs_per_frame).fit_predict(ctable)
        #     print(cluster_ids)

        cluster_dict = {}
        for obj_index in range(len(cluster_ids)):
            cluster_id = cluster_ids[obj_index]
            if cluster_id  == -1:
                continue

            if cluster_id not in cluster_dict: # New cluster found
                cluster_dict[cluster_id] = {
                    'name': [name_list[obj_index]],
                    'pred_labels': [label_list[obj_index]],
                    'score': [score_list[obj_index]],
                    'score_all': [score_all_list[obj_index]],
                    'boxes_lidar': [box_list[obj_index]],
                    'pred_vars': [box_var_list[obj_index]]
                }
            else: # Append to existing cluster
                cluster_dict[cluster_id]['name'].append(name_list[obj_index])
                cluster_dict[cluster_id]['pred_labels'].append(label_list[obj_index])
                cluster_dict[cluster_id]['score'].append(score_list[obj_index])
                cluster_dict[cluster_id]['score_all'].append(score_all_list[obj_index])
                cluster_dict[cluster_id]['boxes_lidar'].append(box_list[obj_index])
                cluster_dict[cluster_id]['pred_vars'].append(box_var_list[obj_index])
        #     print(cluster_dict)

        # Calculate means of each cluster
        final_name_list = []
        final_label_list = []
        final_score_list = []
        final_score_all_list = []
        final_box_list = []
        final_var_list = []
        for cluster_id in cluster_dict:
            cluster = cluster_dict[cluster_id]
            final_name_list.append(cluster['name'][0]) # Take the first one, asssume cluster preds the same class
            final_label_list.append(cluster['pred_labels'][0]) # Take the first one, asssume cluster preds the same class
            final_score_list.append(np.mean(cluster['score']))
            final_score_all_list.append(np.mean(cluster['score_all'], axis=0))
            # NOTE: Get yaw from highest confidence box (we don't take the mean)
            highest_conf_pred_idx = np.argmax(cluster['score']) 
            pred_box_tmp = np.mean(cluster['boxes_lidar'], axis=0)
            pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw
            final_box_list.append(pred_box_tmp)
            # TODO we are not properly combining the variances!
            final_var_list.append(np.mean(cluster['pred_vars'], axis=0))
        final_name_list = np.array(final_name_list)
        final_label_list = np.array(final_label_list)
        final_score_list = np.array(final_score_list)
        final_score_all_list = np.array(final_score_all_list)
        final_box_list = np.array(final_box_list)
        final_var_list = np.array(final_var_list)
        #     print(final_label_list)
        #     print(final_score_list)    
        #     print(final_box_list)
        
        # Add mean output for the frame
        new_pred_dicts.append({
            'frame_id': frame_dict_list[0][0]['frame_id'],
            'name': final_name_list, 
            'pred_labels': final_label_list, 
            'score': final_score_list,
            'score_all': final_score_all_list,
            'boxes_lidar': final_box_list,
            'pred_vars': final_var_list
        })

    return new_pred_dicts
