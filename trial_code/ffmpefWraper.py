# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:36:06 2015

@author: adityanagarajan
"""

import subprocess
#import DFWnet
import os
import shutil

base_dir = '/Users/adityanagarajan/projects/nowcaster/output/prediction_movies'

dirs = os.listdir(base_dir)
dirs = filter(lambda x: x[:3] != '.DS',dirs)


for d in dirs:
#    print base_dir + os.sep + d
    os.chdir(base_dir + os.sep + d )
    subprocess.call(['ffmpeg','-framerate','1','-i','Plot_%d.png','-c:v','libx264','-r','1','-pix_fmt','yuv420p',d + '.mp4'])
#    subprocess.call(['ffmpeg','-i',d + '.mp4' ,'-vf',"setpts=0.2*PTS",d + '_FF.mp4'])
#    print os.getcwd()


    
