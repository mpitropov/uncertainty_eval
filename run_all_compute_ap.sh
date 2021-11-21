#! /usr/bin/env bash

# KITTI PointPillars MODELS
# python compute_average_precision.py KITTI -1 1 pp_openpcdet.pkl
# python compute_average_precision.py KITTI 0 1 pp_mcdropout.pkl
# python compute_average_precision.py KITTI 1 1 pp_ensemble.pkl
# python compute_average_precision.py KITTI 1 0 pp_ensemble_0.pkl # Affirmative
# python compute_average_precision.py KITTI 1 0 pp_ensemble_1.pkl # Affirmative
# python compute_average_precision.py KITTI 1 0 pp_ensemble_2.pkl # Affirmative
# python compute_average_precision.py KITTI 1 0 pp_ensemble_3.pkl # Affirmative
python compute_average_precision.py KITTI 2 1 pp_mimo_a.pkl
# python compute_average_precision.py KITTI 3 1 pp_mimo_b.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c.pkl
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_bs3_0.pkl # Affirmative
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_bs3_1.pkl # Affirmative
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_wide.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_3hds.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_3hds_wide.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_new_params.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_new_params2.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_new_params3.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_new_params4.pkl
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_new_params4_0.pkl # Affirmative
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_new_params4_1.pkl # Affirmative
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_new_params4_run2.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_mcdropout_bs6.pkl
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_mcdropout_bs6_0.pkl # Affirmative
# python compute_average_precision.py KITTI 4 0 pp_mimo_c_mcdropout_bs6_1.pkl # Affirmative

# python compute_average_precision.py KITTI 4 1 pp_mimo_c_bs12_ir.2.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_bs12_ir.1.pkl
# python compute_average_precision.py KITTI 4 1 pp_mimo_c_bs12_ir.05.pkl



# KITTI SECOND MODELS
# python compute_average_precision.py KITTI -1 1 second_openpcdet.pkl
# python compute_average_precision.py KITTI 0 1 second_mcdropout.pkl
# python compute_average_precision.py KITTI 1 1 second_ensemble.pkl
python compute_average_precision.py KITTI 2 1 second_mimo_a.pkl
# python compute_average_precision.py KITTI 4 1 second_mimo_c.pkl


# CADC SECOND MODELS
# python compute_average_precision.py CADC -1 1 cadc_second_baseline.pkl
# python compute_average_precision.py CADC 0 1 cadc_second_mcdropout.pkl
# python compute_average_precision.py CADC 1 1 cadc_second_ensemble.pkl
# python compute_average_precision.py CADC 2 1 cadc_second_mimo_a.pkl
# python compute_average_precision.py CADC 4 1 cadc_second_mimo_c.pkl
# python compute_average_precision.py CADC 4 1 cadc_second_mimo_c_2.pkl
# python compute_average_precision.py CADC 4 1 cadc_second_mimo_c_retrain.pkl
# python compute_average_precision.py CADC 4 1 cadc_second_mimo_c_ir5_br2.pkl

# CADC PointPillars MODELS
# python compute_average_precision.py CADC -1 1 cadc_pp_baseline.pkl
# python compute_average_precision.py CADC 0 1 cadc_pp_mcdropout.pkl
# python compute_average_precision.py CADC 1 1 cadc_pp_ensemble.pkl
# python compute_average_precision.py CADC 4 1 cadc_pp_mimo_c.pkl

