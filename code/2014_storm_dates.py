# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:41:42 2016

@author: adityanagarajan
This script is going to define an array 2014_storm_dates.npy which holds
all the storm dates in 2014 from May-August both inclusive
"""

storm_dates = '''- 128 >> 05/08
- 129 >> 05/09
- 132 >> 05/12
- 133 >> 05/13
- 141 >> 05/21
- 142 >> 05/22
- 143 >> 05/23
- 144 >> 05/24
- 145 >> 05/25
- 146 >> 05/26
- 148 >> 05/28
- 151 >> 05/31
- 160 >> 06/09
- 164 >> 06/13 
- 170 >> 06/19
- 173 >> 06/22
- 174 >> 06/23
- 175 >> 06/24
- 176 >> 06/25
- 179 >> 06/28
- 183 >> 07/02
- 184 >> 07/03
- 195 >> 07/14
- 196 >> 07/15
- 197 >> 07/16
- 198 >> 07/17
- 199 >> 07/18
- 204 >> 07/23
- 205 >> 07/24
- 209 >> 07/28
- 210 >> 07/29
- 211 >> 07/30
- 212 >> 07/31
- 223 >> 08/11
- 228 >> 08/16
- 229 >> 08/17
- 230 >> 08/18
- 231 >> 08/19
- 241 >> 08/29'''
'''
- Add the following dates

    - May 28th
    - May 21st and 22nd
    - May 13th
    - No need for june 18th 2014 at all
    - June 23
    - June 24th
    - June 28th
    - July 2nd, 3rd
    - July 14th
    - july 23 and 24
    - Aug 8,9
'''
import re
import numpy as np
import DFWnet

def main():
    DFW = DFWnet.CommonData()
    print storm_dates
    dates_list = storm_dates.split('\n')
    num_dates =  len(dates_list)
    storm_dates_2014 = np.zeros((num_dates,4))
    ctr=0
    for d in dates_list:
        k = re.findall('\d+',d)
        print k
        DFW.date2doy(14,int(k[1]),int(k[2]))
        storm_dates_2014[ctr,:] = np.array((DFW.doy,14,int(k[1]),int(k[2])))
        ctr+=1
    print storm_dates_2014
    np.save('../data/storm_dates_2014.npy',storm_dates_2014)
        
    
    
if __name__ == '__main__':
    main()