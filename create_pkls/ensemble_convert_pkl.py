import pickle, sys

pkl_file = sys.argv[1]
output_file = sys.argv[2]

ret_dict_list = pickle.load( open( pkl_file, "rb" ) )

# Split a file like this
# listof( listof( dict ) )
# Into individual lists with listof( dict ) of len 1
for ensemble_id in range(len(ret_dict_list[0])):
    new_ret_dict_list = []
    for frame_id in range(len(ret_dict_list)):
        print('reading frame', ret_dict_list[frame_id][ensemble_id]['frame_id'])
        new_ret_dict_list.append([ret_dict_list[frame_id][ensemble_id]])
    pickle.dump( new_ret_dict_list, open( output_file + '_' + str(ensemble_id) + '.pkl', "wb" ) )
