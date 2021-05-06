import pickle, sys

pkl_file = sys.argv[1]
output_file = sys.argv[2]

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
    proper_list = []
    for out in list_of_head_outputs:
        proper_list.append(out)
    new_ret_dict_list.append(proper_list)

pickle.dump( new_ret_dict_list, open( output_file, "wb" ) )
