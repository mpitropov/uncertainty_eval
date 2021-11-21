#! /usr/bin/env bash

# arg 1 is pkl file name
# arg 2 is model type
# arg 3 is cluster type

# Compute the classification score thresholds
python compute_cls_score_threshold.py $1 $3 $4 softmax $2

# Calibrate the model with the classification score thresholds filtering
python calibrate_network.py $1 $3 $4 $2

# Compute the classification score thresholds with the temperature scaling
python compute_cls_score_threshold.py $1 $3 $4 temp_scaled $2

# Final scoring rules and calibration error calculation
python uncertainty_eval.py $1 $3 $4 $2
