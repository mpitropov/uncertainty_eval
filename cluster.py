import numpy as np
from sklearn.cluster import DBSCAN
from scipy.special import softmax

from detection_eval import DetectionEval

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
def cluster_preds(pred_dicts, MIN_CLUSTER_SIZE):
    num_frames = len(pred_dicts)
    num_outputs_per_frame = len(pred_dicts[0])

    # If output is already one dict per frame
    # Only should apply softmax
    if isinstance(pred_dicts[0], dict):
        # If original model then just return the predictions
        if 'score_all' not in pred_dicts[0]:
            return pred_dicts

        for i in range(len(pred_dicts)):
            for j in range(len(pred_dicts[i]['score_all'])):
                pred_dicts[i]['score_all'][j] = softmax(pred_dicts[i]['score_all'][j])
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
            names_one_frame = single_data_dict['name']
            labels_one_frame = get_labels(single_data_dict)
            score_one_frame = single_data_dict['score']
            score_all_one_frame = single_data_dict['score_all']
            boxes_one_frame = single_data_dict['boxes_lidar']
            box_vars_one_frame = single_data_dict['pred_vars']
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

        ctable = DetectionEval.compute_ctable(box_list, box_list, criterion='iou')
        # IDs >= 0 are valid clusters while -1 means the cluster did not reach min samples
        cluster_ids = DBSCAN(eps=0.7, min_samples=MIN_CLUSTER_SIZE).fit_predict(ctable)

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
            # Each variance output in the model is correct for only one output
            # Ex. width variance = width^2 * encoded width variance
            # Instead of width^2 we need to use E^2[width] for a cluster
            # Therefore we must divide the variance by width^2 for that box
            # followed by mutiplying by the mean cluster width^2
            for box_id in range(len(cluster['pred_vars'])):
                # First divide by [w|l|h]^2 for this box to get back encoded variance
                cluster['pred_vars'][box_id][3] /= np.square(cluster['boxes_lidar'][box_id][3])
                cluster['pred_vars'][box_id][4] /= np.square(cluster['boxes_lidar'][box_id][4])
                cluster['pred_vars'][box_id][5] /= np.square(cluster['boxes_lidar'][box_id][5])
                # Now multiply by the expected value (mean) squared to get the correct variance
                cluster['pred_vars'][box_id][3] *= np.square(pred_box_tmp[3])
                cluster['pred_vars'][box_id][4] *= np.square(pred_box_tmp[4])
                cluster['pred_vars'][box_id][5] *= np.square(pred_box_tmp[5])
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
            'frame_id': frame_dict_list[0]['frame_id'],
            'name': final_name_list, 
            'pred_labels': final_label_list, 
            'score': final_score_list,
            'score_all': final_score_all_list,
            'boxes_lidar': final_box_list,
            'pred_vars': final_var_list
        })

    return new_pred_dicts
