#! /usr/bin/env bash

# KITTI PointPillars MODELS
# bash run_uncertainty_eval.sh KITTI pp_mcdropout.pkl 0 1

# bash run_uncertainty_eval.sh KITTI pp_ensemble.pkl 1 1
# bash run_uncertainty_eval.sh KITTI pp_ensemble_0.pkl 1 0
# bash run_uncertainty_eval.sh KITTI pp_ensemble_1.pkl 1 0
# bash run_uncertainty_eval.sh KITTI pp_ensemble_2.pkl 1 0
# bash run_uncertainty_eval.sh KITTI pp_ensemble_3.pkl 1 0

bash run_uncertainty_eval.sh KITTI pp_mimo_a.pkl 2 1

# # # bash run_uncertainty_eval.sh KITTI pp_mimo_b.pkl 2 1

# bash run_uncertainty_eval.sh KITTI pp_mimo_c.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_bs3_0.pkl 2 0
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_bs3_1.pkl 2 0

# bash run_uncertainty_eval.sh KITTI pp_mimo_c_wide.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_3hds.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_3hds_wide.pkl 2 1
# # # bash run_uncertainty_eval.sh KITTI pp_mimo_c_new_params.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_new_params2.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_new_params3.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_new_params4.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_new_params4_run2.pkl 2 1

# bash run_uncertainty_eval.sh KITTI pp_mimo_c_mcdropout_bs6.pkl 2 0
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_mcdropout_bs6_0.pkl 2 0
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_mcdropout_bs6_1.pkl 2 0

# bash run_uncertainty_eval.sh KITTI pp_mimo_c_bs12_ir.2.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_bs12_ir.1.pkl 2 1
# bash run_uncertainty_eval.sh KITTI pp_mimo_c_bs12_ir.05.pkl 2 1


# KITTI SECOND MODELS
# bash run_uncertainty_eval.sh KITTI second_mcdropout.pkl 0 1
# bash run_uncertainty_eval.sh KITTI second_ensemble.pkl 1 1
bash run_uncertainty_eval.sh KITTI second_mimo_a.pkl 2 1
# bash run_uncertainty_eval.sh KITTI second_mimo_c.pkl 2 1

# CADC SECOND MODELS
# bash run_uncertainty_eval.sh CADC cadc_second_mcdropout.pkl 0 1
# bash run_uncertainty_eval.sh CADC cadc_second_ensemble.pkl 1 1
# bash run_uncertainty_eval.sh CADC cadc_second_mimo_c_retrain.pkl 2 1
