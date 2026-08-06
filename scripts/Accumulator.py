#!/usr/bin/env python3
"""
Things we did to improve CPU usage and make voxelling more efficient:
-Floor, then add 0.5 to the voxel before scaling by voxel size, better than rounding, probably faster too

-Downsampled inside of callback, with cloud_down()

-Downsampling uses unique more efficently: unique works better with 1d arrays, 
so a int64 key for each xyz is made with shifting, then using unique

-cloud_down() is used in callback, and also is used before publishing and after bounding
this allows publish only unique points, and also to save unique DOWNSAMPLED points from each message,
lessening the amt of each message needed to be processed, and less published points without losing data

-rospy.timer does a callback every so often, so you can do cloud processing, conditional then publish when possible 

-new make point cloud function, old one was slow, parsing through numpy arrays, to populate fields, and this one turned it
into a contiguous array, indexed specific bits to fill out points, which is faster, makes sense a bit more lower level and being a c operation
contiguous array used when using bits indexing information of a previously numpy/python array

-point_cloud2.read_points(), the point_cloud2 array function was replaced with cloud_to_xyz(), which directly reads x y z to the correct binary position
in the incoming stream of message data, returning concatenated numpy array, instead of using nested for loops to parse through the whoole point cloud, populates and creating long generator
"""
import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs import point_cloud2
import threading
import math
import numpy as np
from collections import deque 
from common import *

class Accumulator:
    def __init__(self): 
        self.lock = threading.Lock()

        #Odom var to hold the x,y,z odom data
        self.latest_pos = None

        #Var to cloud pt cloud
        self.latest_cloud = None

        #Intializing publisher
        self.pub = rospy.Publisher("/Cum_Cloud", PointCloud2, queue_size = 10)

        #Intializing subscriber
        self.sub_o = rospy.Subscriber("/Odometry", Odometry, self.odom_cb, queue_size = 10)

        #Intializing subscriber
        self.sub_c = rospy.Subscriber("/cloud_registered", PointCloud2, self.cloud_cb, queue_size = 10)

        #Flag to see if there is available odom data
        self.is_odom = False

        #For make cloud
        self.voxel_size = 0.067 #Width of voxels
        self.numframes = 10 # how long temporal window lasts in seconds * 10, how many saved clouds
        self.cloud_list = deque(maxlen = self.numframes) #List with downsampled cloud entries
        self.radius = 20 #Bounding radius
        self.cloud_array = None  
        self.voxel_array = None
        self.bounded = None

        #Cum Cloud
        self.cum_cloud = None

        #Timer
        rospy.Timer(rospy.Duration(.1), self.on_timer)  # 10 Hz

#-------Subscriber Thread--------
    def odom_cb(self, msg):
        """!@see ourNode.ourNode.odom_cb"""
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            # print("HERE is x: ", self.latest_pos.x)
            # print("HERE is y: ", self.latest_pos.y, "\n")
            #print("INSIDE HERE IS P: ", self.latest_pos)
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        """!@brief Callback function for the /cloud_registered topic
            @details Updates the latest point cloud data using cloud_to_xyz(). Appends a downsampled version of the point cloud by calling down_cloud() to a deque list of max len 10 (FIFO) for make_cloud().
            @note self.lock is used to ensure publishing thread using cloud data don't get partial data, as this callback is in a separate thread
            @param msg The PointCloud2 message received from the /cloud_registered topic
            @see cloud_to_xyz @see down_cloud @see make_cloud"""

        self.latest_cloud = cloud_to_xyz(msg) 

        reduced = self.down_cloud(self.latest_cloud) #Downsample outside lock to avoid thread being locked excessively

        with self.lock:
            if len(self.latest_cloud):
                self.cloud_list.append(reduced)
        

#-------Publisher Thread--------
    def on_timer(self,event):
        """!@brief Timer callback function that is called every 0.1 seconds. Calls make_cloud and publ.
            @details This func checks if there is odometry data and if numpy array from make_cloud was populated.
            @param event An object of TimerEvent, automatically created every time rospy.Timer fires.
            @see odom_cb @see make_cloud @see publ"""
        #if odometry and unique cloud window is open, then publish
        if self.is_odom and self.make_cloud():
            self.publ()

    def make_cloud(self):
        """!@brief Creates a cumulative point cloud from the downsampled point clouds in the cloud_list deque
            @details The down sampled clouds in cloud_list are bounded by a radius.
            @return 1 if the cumulative point cloud is not empty, 0 otherwise
            @see down_cloud"""
        #reshapes the voxeled cloud deque, so that each cloud in the deque(num frames long) accessed/parsed easier
        self.cloud_array = np.vstack(self.cloud_list)

        disx = self.cloud_array[:,0] - self.latest_pos.x
        disy = self.cloud_array[:,1] - self.latest_pos.y
        disz = self.cloud_array[:,2] - self.latest_pos.z

        #norm of x y z of each coord
        xyzdist = np.column_stack((disx,disy,disz))
        norms = np.linalg.norm(xyzdist, axis = 1) 
        # bounded = cloud_array indexed with boolean mask to save only bounded cloud points
        self.bounded = self.cloud_array[norms < self.radius]
        #use down_cloud not republish repeated voxels
        self.voxel_array = self.down_cloud(self.bounded)   # this is below vstack, as its done after bounding to lessen how many thigns go into down_cloud
        #if voxel array, is not empty, return 1, else return 0
        #in other words, return 1 unless there is no occupied voxels, return 0
        #saves us from publishing empty map, which would cause error at worst and waste memory at best
        if len(self.voxel_array):
            return 1
        return 0

    def publ(self):
        """!@brief Publishes the cumulative bounded point cloud to the /Cum_Cloud topic
            @details Creates a PointCloud2 message using make_pointcloud2_xyz32().
            @see common.make_pointcloud2_xyz32"""
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        self.cum_cloud = make_pointcloud2_xyz32(header, self.voxel_array)
        self.pub.publish(self.cum_cloud)


#-------Helper funcs--------
    def down_cloud(self, cloud):
        """!@brief Downsamples a point cloud using voxelization
            @details Quick Summary of the Alogrithm: 
            1) Place each point in the cloud inside of a integer "bin" number by dividing each of it's coordinates by voxel_size and flooring the result. 
            2) Make all bin numbers in the numpy array to be positive to avoid using 2's complement
            3) Pass a key containing the shifted pos bin numpy array to np.unique() to get the indices of the unique voxel bins
            @note The bit shifting operation << is vectorized for a numpy array, performing it element wise. This means each pos bin coordinate (x, y, or z) in the array is shifted without np.unique() having to compare rows
            , making it faster. Additionally, the | operation is vectorized too.
            @note CPU usage is droppped from 39.5% to 7% using the alogirthm above
            @param A point cloud as a numpy array
            @return A numpy array of shape (N, 3) containing the XYZ coordinates of the downsampled point cloud"""
        
        #This get's the bin num, for assigning bin indexes to each downsampled voxel box
        bins = np.floor( cloud / self.voxel_size).astype(np.int64) 

        #This turns all the bin num to pos, by sub most neg xyz pt in cloud (found w/ axis = 0)
        pos_bins = bins - bins.min(axis=0)

        #Key is a 64 bit int, it pos bins of all cloud pts in order: x, y, z. Uses pos bin so no need for 2s cmp or any goofy stuff
        key = (pos_bins[:, 0] << 42) | (pos_bins[:,1] << 21) | pos_bins[:, 2] 

        #Only grabbing indices of the unique pos x,y,z in 1D key (np.unique beta w/). Note these alr been voxeled
        _, first = np.unique(key, return_index = True) 

        #Now we have the indices of the unique voxel bins b/c first collerates to rows, we now use the og array of voxel bins, scaling by voxel size. 
        voxel_cen = (bins[first] + 0.5) * self.voxel_size #Adding +0.5 to move from corner voxel box to center

        #With using this line below instead, CPU usage is 39.5% verus 7% with the code above
        # voxel_cen = self.voxel_size * np.unique(np.round(cloud/ self.voxel_size), axis=0)

        return voxel_cen

    def run(self):
        rospy.spin()


def main():
    rospy.init_node("Accumulator") #Make the node

    Accumulator().run()

    # rospy.spin()

if __name__ =="__main__":
    main()