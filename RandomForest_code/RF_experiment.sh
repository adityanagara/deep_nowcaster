#!/bin/bash
python RF_prediction_experiments.py False 60 ipw_refl 200 RF_60prediction_ipw_refl_experiment.pkl
python RF_prediction_experiments.py False 60 ipw_refl 400 RF_60prediction_ipw_refl_experiment.pkl
python RF_prediction_experiments.py False 60 ipw_refl 500 RF_60prediction_ipw_refl_experiment.pkl
#python RF_prediction_experiments.py False 60 ipw_refl 900 RF_60prediction_ipw_refl_experiment.pkl
python RF_prediction_experiments.py False 60 refl 200 RF_60prediction_refl_experiment.pkl

python RF_prediction_experiments.py False 60 refl 300 RF_60prediction_refl_experiment.pkl
python RF_prediction_experiments.py False 60 refl 400 RF_60prediction_refl_experiment.pkl
python RF_prediction_experiments.py False 60 refl 500 RF_60prediction_refl_experiment.pkl
