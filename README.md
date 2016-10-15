# deep_nowcaster


This repository contains all code required to reproduce results from my Masters Thesis titled "Towards Learning Spatiotemporal Relationships 
from Multiple Weather Variables using Machine Learning Techniques for Precipitation Nowcasting" (link available soon). The first step is to
build the train/test data set of evolving precipitation fields and evolving moisture fields (termed as NIPW - Normalized 
Integrated Precipitable Water) each of which is a 100 x 100 matrix stored as a numpy array for a given timestep t. Before running the script please ensure you download the data required to make these fields using
the from the link [here](http://emmy9.casa.umass.edu/gpsmet/deep_nowcaster/). This file contains the raw radar data from NCDC in NetCDF 
format and the point measurements of IPW (Integrated Precipitable Water Vapor from the 44 GPS stations). The following script plots
all the images for the storm dates in our experiment and also stores the images in a numpy array. 

```python
python reflectivity_ipw_movies.py
```
This would generate inside the data/dataset/YYYY directory the numpy arrays of all the NIPW and the reflectivity fields from the 
storm dates which we define as our training set and also plot the entire training set into the output folder. There are 48 plots per
day (NIPW and reflectivity sampled at 30 minute intervals). The following video gives a visualization of the evolving precipitation
images and the evolving water vapor images. 
<iframe style="display: inline-block;" width="450" height="315" src="https://www.youtube.com/embed/r_LATx7BdUQ" frameborder="0" allowfullscreen></iframe>

The goal of the project is to build a nowcasting system which uses the joint information provided by the evolving precipitation
and NIPW fields to nowcast the future precipitation fields.

The [BuildDataSet.py](https://github.com/adityanagara/deep_nowcaster/blob/master/includes/BuildDataSet.py) 
contains a set of helper functions to build the training and test data set for training the Random Forest 
and the Convolutional Neural Network. Further explanations of how to train these algorithms are provided 
in the respective folders. 
