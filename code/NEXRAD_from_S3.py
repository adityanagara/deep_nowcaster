# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:49:34 2016

@author: adityanagarajan
"""

import numpy as np

import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import pyart.graph
import tempfile
import pyart.io
import boto
from netCDF4 import Dataset

s3conn = boto.connect_s3()
bucket = s3conn.get_bucket('noaa-nexrad-level2',validate=False)
s3key = bucket.get_key('2015/05/15/KVWX/KVWX20150515_080737_V06.gz')
print s3key

localfile = tempfile.NamedTemporaryFile()
s3key.get_contents_to_filename(localfile.name)
radar = pyart.io.read_nexrad_archive(localfile.name)

display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=(9, 12))

plots = [
    # variable-name in pyart, display-name that we want, sweep-number of radar (0=lowest ref, 1=lowest velocity)
    ['reflectivity', 'Reflectivity (dBZ)', 0],
    ['differential_reflectivity', 'Zdr (dB)', 0],
    ['differential_phase', 'Phi_DP (deg)', 0],
    ['cross_correlation_ratio', 'Rho_HV', 0],
    ['velocity', 'Velocity (m/s)', 1],
    ['spectrum_width', 'Spectrum Width', 1]
]

def plot_radar_images(plots):
    ncols = 2
    nrows = len(plots)/2
    for plotno, plot in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, plotno)
        display.plot(plot[0], plot[2], ax=ax, title=plot[1],
             colorbar_label='',
             axislabels=('East-West distance from radar (km)' if plotno == 6 else '', 
                         'North-South distance from radar (km)' if plotno == 1 else ''))
        display.set_limits((-300, 300), (-300, 300), ax=ax)
        display.set_aspect_ratio('equal', ax=ax)
        display.plot_range_rings(range(100, 350, 100), lw=0.5, col='black', ax=ax)
    plt.show()

plot_radar_images(plots)


