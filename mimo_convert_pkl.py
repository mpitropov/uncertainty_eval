import pickle

pkl_file = "/home/matthew/git/cadc_testing/WISEPCDet_MIMO/output/kitti_models/pointpillar_mimo_var/default/eval/epoch_80/val/default/result.pkl"

ret_dict_list = pickle.load( open( pkl_file, "rb" ) )

# Create a new pickle file
# First list contains number of frames
# Second list contains number of models in ensemble or detection heads
# Each dict is a regular output dict for a frame from OpenPCDet
# listof( listof( dict ) )
new_ret_dict_list = []

for ret_dict in ret_dict_list:
    print('reading frame', ret_dict['frame_id'])
    # Get list of return dicts for each detection head
    list_of_head_outputs = ret_dict['post_nms_head_outputs']
    new_ret_dict_list.append(list_of_head_outputs)

pickle.dump( new_ret_dict_list, open( "result_converted.pkl", "wb" ) )
