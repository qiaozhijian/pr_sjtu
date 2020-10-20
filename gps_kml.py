# -*- coding: UTF-8 -*-
import re
import math
import matplotlib.pyplot as plt
import numpy as np

xy_coordinate = []  # 转换后的XY坐标集
def millerToXY (lon, lat):
    """
    经纬度转换为平面坐标系中的x,y 利用米勒坐标系
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    L = 6381372*math.pi*2
    W = L
    H = L/2
    mill = 2.3
    x = float(lon)*math.pi/180
    y = float(lat)*math.pi/180
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
    x = (W/2)+(W/(2*math.pi))*x
    y = (H/2)-(H/(2*mill))*y
    return x,y

f_in = open(r'/home/qzj/Desktop/gpsData')
f_out = open(r'/home/qzj/Desktop/gps_kml','w')
a=f_in.readlines()
for line in a[1:]:
    m=line.split(',')
    lon=m[7]
    lat=m[6]
    alt=m[8]
    if lon=='nan':
        continue
    gpsData = lon+','+lat+','+alt+' '
    f_out.write(gpsData)
    x,y = millerToXY(lon, lat)
    xy_coordinate.append([x,y])
xy_coordinate = np.asarray(xy_coordinate).reshape(-1,2)
plt.plot(xy_coordinate[:,0],xy_coordinate[:,1])#画连线图
plt.scatter(xy_coordinate[:,0],xy_coordinate[:,1])#画散点图
plt.show()
f_in.close()
f_out.close()
