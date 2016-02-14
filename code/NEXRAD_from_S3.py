# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:49:34 2016

@author: adityanagarajan
"""

#import numpy as np
#
#import matplotlib.pyplot as plt
#import numpy.ma as ma
#import numpy as np
#import pyart.graph
#import tempfile
#import pyart.io
#import boto
#
#s3conn = boto.connect_s3()
#bucket = s3conn.get_bucket('noaa-nexrad-level2',validate=False)
#s3key = bucket.get_key('2015/05/15/KVWX/KVWX20150515_080737_V06.gz')
#
#print s3key
def read_nexrad_aws(filename):
    import tempfile
    import urllib
    import os
    import sys
    import pyart
    baseurl = 'https://noaa-nexrad-level2.s3.amazonaws.com/'
    localfile = tempfile.NamedTemporaryFile()
    code = filename[0:4]
    yyyy = filename[4:8]
    mm = filename[8:10]
    dd = filename[10:12]
    url = baseurl + yyyy + '/' + mm + '/' + dd + '/' + code + '/' + filename
    print(url)
    # Web module protocols are different depending on Python 2 vs. Python 3
    if sys.version_info < (3, 0, 0):
        handle = urllib.urlretrieve(url, localfile.name)
        print(handle)
    else:
        import urllib.request
        import shutil
    # Download the file from 'url' and save it locally under 'file_name':
    with handle.request.urlopen(url) as response, open(localfile.name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    os.system('mv '+localfile.name+' '+localfile.name+'.gz')
    os.system('gzip -d '+localfile.name+'.gz')
    radar = pyart.aux_io.radx.read_radx(localfile.name)
    return radar

#Then to read a file you just need to know the name, e.g.,
radar = read_nexrad_aws('KCYS19990614_211854.gz')
print radar
