//
// Created by qzj on 2020/6/24.
//

#ifndef SRC_GLOBAL_H
#define SRC_GLOBAL_H

#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "nav_msgs/Path.h"
#include <sensor_msgs/Imu.h>
#include "std_msgs/Bool.h"
#include "geometry_msgs/PoseStamped.h"
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/UInt8.h"
#include "std_msgs/UInt8MultiArray.h"
#include "std_msgs/Int32.h"
#include <typeinfo>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include<iostream>
#include<string>
#include<sstream>
#include<stdio.h>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

#define SWOP(n) (((n & 0x00FF) << 8 ) | ((n & 0xFF00) >> 8))
#define CAT(a,b) (((a & 0x00FF) << 8 ) | (b & 0x00FF))
#define CAT32(a,b,c,d) (((a & 0x000000FF) << 24) | ((b & 0x000000FF) << 16) | ((c & 0x000000FF) << 8) | ((d & 0x000000FF) << 0))
#define foo(arr) (sizeof(arr)/sizeof(arr[0]))


#define BIT_0(n)  (((n) & (1 << 0))!=0)
#define BIT_1(n)  (((n) & (1 << 1))!=0)
#define BIT_2(n)  (((n) & (1 << 2))!=0)
#define BIT_3(n)  (((n) & (1 << 3))!=0)
#define BIT_4(n)  (((n) & (1 << 4))!=0)
#define BIT_5(n)  (((n) & (1 << 5))!=0)
#define BIT_6(n)  (((n) & (1 << 6))!=0)
#define BIT_7(n)  (((n) & (1 << 7))!=0)

#endif //SRC_GLOBAL_H
