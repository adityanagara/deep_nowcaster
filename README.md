# deep_nowcaster

## Introduction

This repository contains all code required to reproduce results from my Masters Thesis Exploration into [machine learning techniques for precipitation nowcasting](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1501&context=masters_theses_2). As
a byproduct this repo also contains APIs to access GPS RINEX files from FTP database, and ASOS data from NCDC. We also open source
the script to map weather variables to GPS stations. 

## Dependencies

1. numpy 
2. scipy 
3. netCDF
4. scikit-learn
5. cuda
5. theano
6. lasagne

## Build Training and Test dataset

The first step is to build the train/test data set of evolving precipitation fields and evolving moisture fields (termed as NIPW - Normalized 
Integrated Precipitable Water) each of which is a 100 x 100 matrix stored as a numpy array for a given timestep t. Before running the script please ensure you
download the data(~1.7 GB uncompressed) required to make these fields using the from the link [here](http://emmy9.casa.umass.edu/gpsmet/deep_nowcaster/).
This file contains the raw radar data from NCDC in NetCDF format and the point measurements of IPW (Integrated Precipitable Water Vapor from the 44 GPS stations around KFWS
for the years 2014,2015 and 2016). The following script plots all the images for the storm dates in our experiment and also stores the images in a numpy array
inside data/dataset/YYYY. From inside the Preprocessing_code directory run:

```python
python reflectivity_ipw_movies.py
```
48 plots (NIPW and reflectivity sampled at 30 minute intervals) and numpy arrays for each day are generated. The following shows 
example plots of the precipitation fields overlapped over the NIPW fields. The video sequence of the evolving
precipitation fields and NIPW fields can be found in this [youtube video](https://www.youtube.com/watch?v=r_LATx7BdUQ). 

![example plot](https://github.com/adityanagara/deep_nowcaster/blob/master/Preprocessing_code/Plot_43.png)

We explore machine learning techniques which can capture the spatiotemporal relationships between the evolving precipitation fields
and NIPW fields to be able to nowcast precipitation.

## Train and Test models

[BuildDataSet.py](https://github.com/adityanagara/deep_nowcaster/blob/master/includes/BuildDataSet.py) and other scripts 
in the includes directory contains helper functions to build training and validation data sets and calculating performance
metrics, ensure that includes directory is added to your `PYTHON_PATH` variables. 

### Random Forest(RF) Classifier

We train a random forest classifier using a set of features engineered by taking the spatial statistics of the 33 x 33 window of 
points around the pixel point we are predicting. The set of routines in [BuildDataSet.py](https://github.com/adityanagara/deep_nowcaster/blob/master/includes/BuildDataSet.py)
does this for us and creates a dataset ready to be trained by the Random Forest. From inside the RandomForest_code directory, run
the following script:

```python
python RF_prediction_experiments.py True 60 ipw_refl 600 RF_60prediction_ipw_refl_experiment 6
```

which trains a RF classifier with 600 trees in the forest and a max depth of 6 and saves the results in the file `600RF_60prediction_refl_experiment_max_depth6.pkl`. 
The file contains all the performance metrics evaluated in the training and validation set as defined by the class `NOWCAST_performance()` in [ModelMetrics.py](https://github.com/adityanagara/deep_nowcaster/blob/master/includes/ModelMetrics.py). 

### Convolutional Neural Networks(CNN)

Unlike the Random Forest classifier we feed the CNN with the actual 33 x 33 frames around the pixel point as features. The weights of convolution
filters are learnt for each variable at each time step. The following script runs a single layer CNN with separate connections for the precipitation fields
and separate connections for the NIPW fields. From inside the `CNN_code` directory run:

```python
python Deep_NN_nowcasting_experiments.py
```
The program first creates the training and validation dataset inside the directory `data/TrainTest/points/`. We train our CNN using
a Tesla K80 GPU on [MGHPCC](http://www.mghpcc.org/). 

