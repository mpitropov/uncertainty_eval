import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.special import softmax

from detection_eval import DetectionEval

def get_labels(data_dict):
    if 'gt_labels' in data_dict:
        return data_dict['gt_labels']
    if 'name' in data_dict:
        # classes = ['Car', 'Pedestrian', 'Cyclist']
        classes = ['Car', 'Pedestrian', 'Pickup_Truck']
        # classes = ['car','truck', 'construction_vehicle', 'bus', 'trailer', \
                    # 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
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
def cluster_preds(pred_dicts, MIN_CLUSTER_SIZE, tracking_mode=False):
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
        # tracking lists
        if tracking_mode:
            bbox_list = []
            location_list = []
            dimensions_list = []
            rotation_y_list = []
            alpha_list = []

        for single_data_dict in frame_dict_list:
            names_one_frame = single_data_dict['name']
            labels_one_frame = get_labels(single_data_dict)
            score_one_frame = single_data_dict['score']
            score_all_one_frame = single_data_dict['score_all']
            boxes_one_frame = single_data_dict['boxes_lidar']
            box_vars_one_frame = single_data_dict['pred_vars']
            if tracking_mode:
                bbox_one_frame = single_data_dict['bbox']
                location_one_frame = single_data_dict['location']
                dimensions_one_frame = single_data_dict['dimensions']
                rotation_y_one_frame = single_data_dict['rotation_y']
                alpha_one_frame = single_data_dict['alpha']

            for i in range(len(labels_one_frame)):
                name_list.append(names_one_frame[i])
                label_list.append(labels_one_frame[i])
                CALIBRATION_MODE = False
                NO_TEMP_SCALE = True
                if CALIBRATION_MODE:
                    score_list.append(score_one_frame[i])
                    score_all_list.append(score_all_one_frame[i]) # Don't apply softmax
                elif NO_TEMP_SCALE:
                    score_list.append(score_one_frame[i])
                    score_all_list.append(softmax(score_all_one_frame[i])) # Apply softmax
                else:
                    # T_clf = [None, 0.7, 0.3, 0.3]
                    T_clf = [None, 0.7, 0.3, 0.3]
                    T = T_clf[labels_one_frame[i]]
                    score_list.append(max(softmax(score_all_one_frame[i] / T)[:3]))
                    score_all_list.append(softmax(score_all_one_frame[i] / T)) # Apply softmax
                box_list.append(boxes_one_frame[i])
                box_var_list.append(box_vars_one_frame[i])
                if tracking_mode:
                    bbox_list.append(bbox_one_frame[i])
                    location_list.append(location_one_frame[i])
                    dimensions_list.append(dimensions_one_frame[i])
                    rotation_y_list.append(rotation_y_one_frame[i])
                    alpha_list.append(alpha_one_frame[i])

        name_list = np.array(name_list)
        label_list = np.array(label_list)
        score_list = np.array(score_list)
        score_all_list = np.array(score_all_list)
        box_list = np.array(box_list)
        box_var_list = np.array(box_var_list)
        if tracking_mode:
            bbox_list = np.array(bbox_list)
            location_list = np.array(location_list)
            dimensions_list = np.array(dimensions_list)
            rotation_y_list = np.array(rotation_y_list)
            alpha_list = np.array(alpha_list)

        if len(box_list[0]) !=7:
            print('wrong box len')
            print(len(box_list))
            print(len(box_list[0]))
            exit()

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
                if tracking_mode:
                    cluster_dict[cluster_id]['bbox'] = [bbox_list[obj_index]]
                    cluster_dict[cluster_id]['location'] = [location_list[obj_index]]
                    cluster_dict[cluster_id]['dimensions'] = [dimensions_list[obj_index]]
                    cluster_dict[cluster_id]['rotation_y'] = [rotation_y_list[obj_index]]
                    cluster_dict[cluster_id]['alpha'] = [alpha_list[obj_index]]
            else: # Append to existing cluster
                cluster_dict[cluster_id]['name'].append(name_list[obj_index])
                cluster_dict[cluster_id]['pred_labels'].append(label_list[obj_index])
                cluster_dict[cluster_id]['score'].append(score_list[obj_index])
                cluster_dict[cluster_id]['score_all'].append(score_all_list[obj_index])
                cluster_dict[cluster_id]['boxes_lidar'].append(box_list[obj_index])
                cluster_dict[cluster_id]['pred_vars'].append(box_var_list[obj_index])
                if tracking_mode:
                    cluster_dict[cluster_id]['bbox'].append(bbox_list[obj_index])
                    cluster_dict[cluster_id]['location'].append(location_list[obj_index])
                    cluster_dict[cluster_id]['dimensions'].append(dimensions_list[obj_index])
                    cluster_dict[cluster_id]['rotation_y'].append(rotation_y_list[obj_index])
                    cluster_dict[cluster_id]['alpha'].append(alpha_list[obj_index])
        #     print(cluster_dict)

        # Calculate means of each cluster
        final_name_list = []
        final_label_list = []
        final_score_list = []
        final_score_all_list = []
        final_shannon_entropy_list = []
        final_aleatoric_entropy_list = []
        final_box_list = []
        final_var_list = []
        final_cluster_size_list = []
        if tracking_mode:
            final_bbox_list = []
            final_location_list = []
            final_dimensions_list = []
            final_rotation_y_list = []
            final_alpha_list = []

        # load the model from disk
        USE_ISO_REG = False
        if USE_ISO_REG:
            filename = 'mimo_a_multiclass_isotonic_reg.sav'
            iso_reg_model = pickle.load(open(filename, 'rb'))

        for cluster_id in cluster_dict:
            cluster = cluster_dict[cluster_id]

            # Merge Mode
            MERGE_MODES = ['max', 'mean', 'wbf', 'var-wbf',  'var-wbf+', 'min-entropy', 'MU', 'entropy-wbf']
            merge_mode = MERGE_MODES[1]

            if merge_mode == 'max':
                highest_conf_pred_idx = np.argmax(cluster['score'])

                final_score_list.append(cluster['score'][highest_conf_pred_idx])
                final_box_list.append(cluster['boxes_lidar'][highest_conf_pred_idx])
                final_var_list.append(cluster['pred_vars'][highest_conf_pred_idx])
                final_name_list.append(cluster['name'][highest_conf_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_conf_pred_idx])
                final_score_all_list.append(cluster['score_all'][highest_conf_pred_idx])
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'min-entropy':
                from scipy import stats
                entropy_list = []
                for i in range(len(cluster['pred_vars'])):
                    entropy_list.append(stats.entropy(cluster['score_all'][i], base = 2))
                lowest_entropy_pred_idx = np.argmin(entropy_list)

                final_score_list.append(cluster['score'][lowest_entropy_pred_idx])
                final_box_list.append(cluster['boxes_lidar'][lowest_entropy_pred_idx])
                final_var_list.append(cluster['pred_vars'][lowest_entropy_pred_idx])
                final_name_list.append(cluster['name'][lowest_entropy_pred_idx])
                final_label_list.append(cluster['pred_labels'][lowest_entropy_pred_idx])
                final_score_all_list.append(cluster['score_all'][lowest_entropy_pred_idx])
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'entropy-wbf':
                highest_conf_pred_idx = np.argmax(cluster['score'])

                from scipy import stats
                entropy_list = []
                for i in range(len(cluster['pred_vars'])):
                    entropy_list.append(stats.entropy(cluster['score_all'][i], base = 2))
                entropy_list = np.array(entropy_list)
                lowest_entropy_pred_idx = np.argmin(entropy_list)

                norm_entropys = (1.0 / entropy_list) / np.sum(1.0 / entropy_list)
                pred_box_tmp = np.average(cluster['boxes_lidar'], axis=0, weights=norm_entropys)
                pred_box_tmp[6] = cluster['boxes_lidar'][lowest_entropy_pred_idx][6] # Replace yaw

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

                mean_cluster_var = np.average(cluster['pred_vars'], axis=0, weights=norm_entropys)
                final_score_list.append(np.average(cluster['score'], weights=norm_entropys))

                final_box_list.append(pred_box_tmp)
                final_var_list.append(mean_cluster_var)

                final_name_list.append(cluster['name'][lowest_entropy_pred_idx])
                final_label_list.append(cluster['pred_labels'][lowest_entropy_pred_idx])
                final_score_all_list.append(np.average(cluster['score_all'], axis=0, weights=norm_entropys))
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'MU': # Mutual Information
                from scipy import stats
                # Predictive Entropy: Captures total uncertainty (H = MI + AE)
                # Calculated as entropy of mean categorical distribution
                pred_entropy = stats.entropy( np.mean(cluster['score_all'], axis=0), base = 2 )
                # Aleatoric Entropy: AE
                # Calculated by averaging the entropy of each sampled categorical distribution
                entropy_list = []
                for i in range(len(cluster['pred_vars'])):
                    entropy_list.append(stats.entropy(cluster['score_all'][i], base = 2))
                aleatoric_entropy = np.mean(entropy_list)
                # Mutual Information: MI = H - AE
                # mutual_info = pred_entropy - aleatoric_entropy
                mutual_info = entropy_list - aleatoric_entropy

                highest_entropy_pred_idx = np.argmax(mutual_info)

                final_score_list.append(cluster['score'][highest_entropy_pred_idx])
                final_box_list.append(cluster['boxes_lidar'][highest_entropy_pred_idx])
                final_var_list.append(cluster['pred_vars'][highest_entropy_pred_idx])
                final_name_list.append(cluster['name'][highest_entropy_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_entropy_pred_idx])
                final_score_all_list.append(cluster['score_all'][highest_entropy_pred_idx])
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'mean':
                # # Temp scaling
                # T_vals = [
                #     [0.606, 0.803, 0.003, 0.618, 0.596, 0.913, 3.0], # Car
                #     [0.715, 0.920, 0.016, 1.309, 0.500, 0.958, 0.203], # Ped
                #     [0.500, 0.510, 0.016, 0.538, 0.500, 0.581, 0.250] # Cyc
                # ]
                # for i in range(len(T_vals[0])):
                #     for obj_id in range(len(cluster['pred_vars'])):
                #         tmp_T = T_vals[cluster['pred_labels'][obj_id] - 1] # Convert to list index
                #         cluster['pred_vars'][obj_id][i] = cluster['pred_vars'][obj_id][i] / tmp_T[i]

                from scipy.stats import circmean
                # blah = False
                # for i in range(len(cluster['pred_labels']) - 1):
                #     if cluster['pred_labels'][i] != cluster['pred_labels'][i+1]:
                #         print('mismatch label in cluster')
                #         break
                #     # angle_a = cluster['boxes_lidar'][i][6]
                #     # if angle_a > np.pi:
                #     #     angle_a -= 2*np.pi
                #     # angle_b = cluster['boxes_lidar'][i+1][6]
                #     # if angle_b > np.pi:
                #     #     angle_b -= 2*np.pi
                #     # if not np.isclose(angle_a, angle_b, atol=0.4):
                #     #     print('yaw mismatch', angle_a, angle_b)
                #     #     break
                #     # We must use circmean for edge case angles
                #     angle_a = cluster['boxes_lidar'][i][6]
                #     angle_b = cluster['boxes_lidar'][i+1][6]
                #     circmean_of_angles = circmean([angle_a, angle_b], high = np.pi, low = -np.pi)
                #     testinga = circmean([angle_a], high = np.pi, low = -np.pi)
                #     testingb = circmean([angle_b], high = np.pi, low = -np.pi)
                #     if np.abs(testinga - circmean_of_angles) > 1.0 and np.abs(testingb - circmean_of_angles) > 1.0:
                #         print("circ mean test")
                #         for idx in range(len(cluster['boxes_lidar'])):
                #             print('angle in cluster', cluster['boxes_lidar'][idx][6])
                #         print("pred_boxes angle_a", angle_a)
                #         print("pred_boxes angle_b", angle_b)
                #         print("real mean of angles", circmean_of_angles)
                #         print("fake mean of angles", testinga)
                #         print("fake mean of angles", testingb)
                #         for j in range(len(cluster['pred_labels'])):
                #             final_score_list.append(cluster['score'][j])
                #             final_box_list.append(cluster['boxes_lidar'][j])
                #             final_var_list.append(cluster['pred_vars'][j])
                #             final_name_list.append(cluster['name'][j])
                #             final_label_list.append(cluster['pred_labels'][j])
                #             final_score_all_list.append(cluster['score_all'][j])
                #             final_cluster_size_list.append(1)
                #             final_shannon_entropy_list.append(stats.entropy(cluster['score_all'][j]))
                #             final_aleatoric_entropy_list.append(stats.entropy(cluster['score_all'][j]))
                #             if tracking_mode:
                #                 final_bbox_list.append(cluster['bbox'][j])
                #                 final_location_list.append(cluster['location'][j])
                #                 final_dimensions_list.append(cluster['dimensions'][j])
                #                 final_rotation_y_list.append(cluster['rotation_y'][j])
                #                 final_alpha_list.append(cluster['alpha'][j])
                #         blah = True
                #         continue
                # if blah == True:
                #     continue
                from scipy import stats
                # Predictive Entropy: Captures total uncertainty (H = MI + AE)
                # Calculated as entropy of mean categorical distribution
                pred_entropy = stats.entropy( np.mean(cluster['score_all'], axis=0), base = 2 )
                # Aleatoric Entropy: AE
                # Calculated by averaging the entropy of each sampled categorical distribution
                entropy_list = []
                for i in range(len(cluster['pred_vars'])):
                    entropy_list.append(stats.entropy(cluster['score_all'][i], base = 2))
                aleatoric_entropy = np.mean(entropy_list)
                final_shannon_entropy_list.append(pred_entropy)
                final_aleatoric_entropy_list.append(aleatoric_entropy)

                highest_conf_pred_idx = np.argmax(cluster['score'])

                pred_box_tmp = np.mean(cluster['boxes_lidar'], axis=0)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw

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

                mean_cluster_var = np.mean(cluster['pred_vars'], axis=0)
                if USE_ISO_REG:
                    # final_score_list.append( iso_reg_model.predict([np.mean(cluster['score'])])[0] )
                    prob = iso_reg_model.predict_proba([np.mean(cluster['score_all'], axis=0)])[0]
                    final_score_list.append( prob[np.argmax(prob[:3])])
                    final_score_all_list.append(prob)
                else:
                    final_score_list.append(np.mean(cluster['score']))
                    final_score_all_list.append(np.mean(cluster['score_all'], axis=0))

                final_box_list.append(pred_box_tmp)
                final_var_list.append(mean_cluster_var)

                final_name_list.append(cluster['name'][highest_conf_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_conf_pred_idx])
                final_cluster_size_list.append(len(cluster['pred_vars']))
                if tracking_mode:
                    final_bbox_list.append(np.mean(cluster['bbox'], axis=0))
                    final_location_list.append(np.mean(cluster['location'], axis=0))
                    final_dimensions_list.append(np.mean(cluster['dimensions'], axis=0))
                    final_rotation_y_list.append(cluster['rotation_y'][highest_conf_pred_idx])
                    final_alpha_list.append(np.mean(cluster['alpha']))
                continue
            elif merge_mode == 'wbf':
                # weighted box fusion: use normalized scores as weights
                highest_conf_pred_idx = np.argmax(cluster['score'])

                norm_scores = cluster['score'] / np.sum(cluster['score'])
                pred_box_tmp = np.average(cluster['boxes_lidar'], axis=0, weights=norm_scores)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw

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

                mean_cluster_var = np.average(cluster['pred_vars'], axis=0, weights=norm_scores)
                final_score_list.append(np.average(cluster['score'], weights=norm_scores))

                final_box_list.append(pred_box_tmp)
                final_var_list.append(mean_cluster_var)

                final_name_list.append(cluster['name'][highest_conf_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_conf_pred_idx])
                final_score_all_list.append(np.average(cluster['score_all'], axis=0, weights=norm_scores))
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'var-wbf':
                # var weighted box fusion: use inverted normalized mean variances as weights
                highest_conf_pred_idx = np.argmax(cluster['score'])

                pred_box_tmp = np.mean(cluster['boxes_lidar'], axis=0)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw

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

                mean_cluster_var_per_obj = np.mean(cluster['pred_vars'], axis=1)
                inv_norm_mean_var = (1.0 / mean_cluster_var_per_obj) / np.sum(1.0 / mean_cluster_var_per_obj)

                pred_box_tmp = np.average(cluster['boxes_lidar'], axis=0, weights=inv_norm_mean_var)
                highest_inv_norm_mean_var = np.argmax(inv_norm_mean_var)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_inv_norm_mean_var][6] # Replace yaw
                mean_cluster_var = np.average(cluster['pred_vars'], axis=0, weights=inv_norm_mean_var)

                final_score_list.append(np.average(cluster['score'], weights=inv_norm_mean_var))

                final_box_list.append(pred_box_tmp)
                final_var_list.append(mean_cluster_var)

                final_name_list.append(cluster['name'][highest_conf_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_conf_pred_idx])
                final_score_all_list.append(np.average(cluster['score_all'], axis=0, weights=inv_norm_mean_var))
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue
            elif merge_mode == 'var-wbf+':
                # var weighted box fusion: use inverted normalized mean variances as weights
                highest_conf_pred_idx = np.argmax(cluster['score'])

                pred_box_tmp = np.mean(cluster['boxes_lidar'], axis=0)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw

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

                # Test adding ensemble variance to predicted variances
                mean_cluster_var_per_obj = np.mean(cluster['pred_vars'] + np.var(cluster['boxes_lidar'], axis=0), axis=1)
                inv_norm_mean_var = (1.0 / mean_cluster_var_per_obj) / np.sum(1.0 / mean_cluster_var_per_obj)

                pred_box_tmp = np.average(cluster['boxes_lidar'], axis=0, weights=inv_norm_mean_var)
                highest_inv_norm_mean_var = np.argmax(inv_norm_mean_var)
                pred_box_tmp[6] = cluster['boxes_lidar'][highest_inv_norm_mean_var][6] # Replace yaw
                mean_cluster_var = np.average(cluster['pred_vars'], axis=0, weights=inv_norm_mean_var)

                final_score_list.append(np.average(cluster['score'], weights=inv_norm_mean_var))

                final_box_list.append(pred_box_tmp)
                final_var_list.append(mean_cluster_var)

                final_name_list.append(cluster['name'][highest_conf_pred_idx])
                final_label_list.append(cluster['pred_labels'][highest_conf_pred_idx])
                final_score_all_list.append(np.average(cluster['score_all'], axis=0, weights=inv_norm_mean_var))
                final_cluster_size_list.append(len(cluster['pred_vars']))
                continue

            # NOTE: Get yaw from highest confidence box (we don't take the mean)
            highest_conf_pred_idx = np.argmax(cluster['score']) 
            # pred_box_tmp = np.mean(cluster['boxes_lidar'], axis=0)
            # pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw
            softmax_summed_vars = softmax(1.0 - softmax( np.sum(cluster['pred_vars'], axis=1) ))
            pred_box_tmp = np.average(cluster['boxes_lidar'], axis=0, weights=softmax_summed_vars)
            pred_box_tmp[6] = cluster['boxes_lidar'][highest_conf_pred_idx][6] # Replace yaw

            # pred_var_a = np.sum(cluster['pred_vars'][0])
            # pred_var_b = np.sum(cluster['pred_vars'][1])
            # if np.abs(pred_var_a - pred_var_b) > 0.1:
            #     print('large difference in variances')
            #     # print(cluster['pred_vars'])
            #     print('A and B', pred_var_a, pred_var_b)
            #     summed_vars = np.sum(cluster['pred_vars'], axis=1)
            #     print('stuff', summed_vars)
            #     print('stuff', softmax(summed_vars) )

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

            mean_cluster_var = np.mean(cluster['pred_vars'], axis=0)
            var_means = np.var(cluster['boxes_lidar'], axis=0)
            # mean_cluster_var += var_means
            # if np.sum(var_means) > 0.5:
            #     print('cluster box means', cluster['boxes_lidar'])
            #     print('variances of means', var_means)
            #     print('mean box', pred_box_tmp)

            #     print('cluster box vars', cluster['pred_vars'])
            #     print('vars mean', mean_cluster_var)
            #     exit()

            # Skip high variance objects
            tmp_label = cluster['pred_labels'][0]
            var_sum = np.sum(mean_cluster_var)

            # mean_score = np.mean(cluster['score'])
            mean_score = np.average(cluster['score'], axis=0, weights=softmax_summed_vars)

            import math
            dist = math.sqrt(pred_box_tmp[0] ** 2 + pred_box_tmp[1] ** 2)
            MAX_DIST = 80.0
            from scipy.special import entr
            obj_entropy = entr( np.mean(cluster['score_all'], axis=0) ).sum()
            MAX_ENTROPY = 0.7

            # if tmp_label == 1:
            #     MAX_VAR_CAR = 0.65 + 1000
            #     if var_sum > MAX_VAR_CAR: # Car
            #         continue
            #     else:
            #         MAX_VAR_CAR = 0.65
            #         new_score = mean_score + (-0.2*(var_sum/MAX_VAR_CAR) + 0.1)
            # elif tmp_label == 2:
            #     MAX_VAR_PED = 1.40 + 1000
            #     if var_sum > MAX_VAR_PED: # Ped
            #         continue
            #     else:
            #         MAX_VAR_PED = 1.40
            #         new_score = mean_score + (-0.2*(var_sum/MAX_VAR_PED) + 0.1)
            # elif tmp_label == 3:
            #     MAX_VAR_CYC = 0.40 + 1000
            #     if var_sum > MAX_VAR_CYC: # Cyc
            #         continue
            #     else:
            #         MAX_VAR_CYC = 0.40
            #         new_score = mean_score + (-0.2*(var_sum/MAX_VAR_CYC) + 0.1)

            # new_score += (-0.1*(dist/MAX_DIST) + 0.05)
            # new_score += (-0.1*(obj_entropy/MAX_ENTROPY) + 0.05)

            # if new_score < 0.10:
                # continue

            # final_score_list.append(max(min(new_score, 1.0), 0.0))

            # Use DT to prune FPs
            from scipy.special import entr

            # obj_entropy = entr( np.mean(cluster['score_all'], axis=0) ).sum()
            # if normal_mode:
            #     x_val = [np.concatenate(([cluster['pred_labels'][0], mean_score, dist], [obj_entropy]))]
            # else:
            #     x_val = [np.concatenate(([cluster['pred_labels'][0], mean_score, dist, np.sum(mean_cluster_var)], [obj_entropy]))]
            # # if normal_mode:
            # #     x_val = [np.concatenate(([cluster['pred_labels'][0], mean_score], pred_box_tmp))]
            # # else:
            # #     x_val = [np.concatenate(([cluster['pred_labels'][0], mean_score], pred_box_tmp, mean_cluster_var))]
            # y_val = clf.predict(x_val)
            # # print('prediction class', y_val)
            # y_val = clf.predict_proba(x_val)
            # print('proba', y_val)
            # exit()
            # if y_val[0] == 0:
            #     y_val = clf.predict_proba(x_val)
            #     print('Predicted FP with confidence', y_val)
            #     if y_val[0][0] > 0.99:
            #         continue

            # if mean_score < 0.12:
            #     continue
            final_score_list.append(mean_score)

            final_box_list.append(pred_box_tmp)
            final_var_list.append(mean_cluster_var)

            final_name_list.append(cluster['name'][0]) # Take the first one, asssume cluster preds the same class
            final_label_list.append(cluster['pred_labels'][0]) # Take the first one, asssume cluster preds the same class
            final_score_all_list.append(np.mean(cluster['score_all'], axis=0))
            final_cluster_size_list.append(len(cluster['pred_vars']))

            if tracking_mode:
                final_bbox_list.append(np.mean(cluster['bbox'], axis=0))
                final_location_list.append(np.mean(cluster['location'], axis=0))
                final_dimensions_list.append(np.mean(cluster['dimensions'], axis=0))
                final_rotation_y_list.append(cluster['rotation_y'][highest_conf_pred_idx])
                final_alpha_list.append(np.mean(cluster['alpha']))

        if len(final_name_list) == 0:
            print('EMPTY FRAME DETECTED')
            # final_score_list.append(0)
            # final_box_list.append([1,1,1,1,1,1,1])
            # final_var_list.append([0.01,0.01,0.01,0.01,0.01,0.01,0.01])
            # final_name_list.append('Car')
            # final_label_list.append(1)
            # final_score_all_list.append([0,0,0,1.0])
            # final_shannon_entropy_list.append(1.0)
            # final_aleatoric_entropy_list.append(1.0)
            # final_cluster_size_list.append(1)

        final_name_list = np.array(final_name_list)
        final_label_list = np.array(final_label_list)
        final_score_list = np.array(final_score_list)
        final_score_all_list = np.array(final_score_all_list)
        final_shannon_entropy_list = np.array(final_shannon_entropy_list)
        final_aleatoric_entropy_list = np.array(final_aleatoric_entropy_list)
        final_box_list = np.array(final_box_list)
        final_var_list = np.array(final_var_list)
        final_cluster_size_list = np.array(final_cluster_size_list)
        if tracking_mode:
            final_bbox_list = np.array(final_bbox_list)
            final_location_list = np.array(final_location_list)
            final_dimensions_list = np.array(final_dimensions_list)
            final_rotation_y_list = np.array(final_rotation_y_list)
            final_alpha_list = np.array(final_alpha_list)
        
        # Add mean output for the frame
        seq_id = None
        if tracking_mode:
            seq_id = frame_dict_list[0]['seq_id']
        
        new_pred_dict = {
            'frame_id': frame_dict_list[0]['frame_id'],
            'seq_id': seq_id,
            'name': final_name_list, 
            'pred_labels': final_label_list, 
            'score': final_score_list,
            'score_all': final_score_all_list,
            'shannon_entropy': final_shannon_entropy_list,
            'aleatoric_entropy': final_aleatoric_entropy_list,
            'boxes_lidar': final_box_list,
            'pred_vars': final_var_list,
            'cluster_size': final_cluster_size_list
        }
        if tracking_mode:
            new_pred_dict['bbox'] = final_bbox_list
            new_pred_dict['location'] = final_location_list
            new_pred_dict['dimensions'] = final_dimensions_list
            new_pred_dict['rotation_y'] = final_rotation_y_list
            new_pred_dict['alpha'] = final_alpha_list
        new_pred_dicts.append(new_pred_dict)

    return new_pred_dicts
