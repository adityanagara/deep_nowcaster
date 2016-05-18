# deep_nowcaster

## Objective 

The objective of this project is to develop a nowcasting system (weather prediction system) by integrating 
data from various different sources. This git hub repo consists of all the code required to build the data pipline for
my Masters Thesis titled "Exploration Into Machine Learning techniques for precipitation nowcasting" advised by
David L. Pepyne. This is currently a work in progress and thus this readme document will only detail a very high level 
description of whats in here. 

## Data Sources 

The domain of our predictions is the Dallas Fort-Worth area Texas. 

## Code 

1. The first step is to identify the sensore we are going to extract data from. To do this we are going to pick a radar
site in this case KFWS (DFW Texas lat: 32.57278, long: -97.30278). Then we identify the Continuously Operated GPS Stations
within the 230 km range radius of this radar by parsing thru [GPS station log files](www.ngs.noaa.gov). The output of this 
script should be a csv file named 'KFWS_230km_sites.csv' containing the station name lat, long, height. 

```
cd code
python NEXRAD_GPS_NOAA_SOPAC.py
```

2. Find the weather stations within the KFWS range radius. To do this we parse [ASOS](http://weather.noaa.gov/tg/site.shtml)
station logs to find all weather stations in our domain. This outputs a file gps_wxstation.csv. 

```
python WXstations_KFWS.py
```

3. 