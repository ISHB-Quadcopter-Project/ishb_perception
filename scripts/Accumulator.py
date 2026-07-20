#!/usr/bin/env python3
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
        self.voxel_size = 0.067
        self.numframes = 50 # how long temporal window lasts in seconds * 10, how many saved clouds
        self.cloud_list = deque(maxlen = self.numframes)
        self.radius = 20
        self.cloud_array = None
        self.voxel_array = None

        #Cum Cloud
        self.cum_cloud = None

    def publ(self):
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        self.cum_cloud = point_cloud2.create_cloud_xyz32(header, self.voxel_array)
        self.pub.publish(self.cum_cloud)
        
        #TODO call the make_cloud func and publish

    def odom_cb(self, msg):
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            #print("INSIDE HERE IS P: ", self.latest_pos)
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        self.latest_cloud = np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y", "z"), skip_nans = True))) # puts the message x and y values into latest_cloud n x 2 rows
        # print("one cloud:  ", self.latest_points, "\n")
        # print("sh of one cloud: ", np.shape((self.latest_points)))
        # print("sh of asdfasdfcloud: ", np.shape(self.latest_points[0,:]))
        if len(self.latest_cloud) : self.cloud_list.append(self.latest_cloud) # this adds to cloud_list, the deque, list of n x 2 rows/coords 
            

    def make_cloud(self, odom):
        # print("DIST ODOM: ", odom, "\n")
        Odomx = odom.x
        Odomy = odom.y
        Odomz = odom.z

        if len(self.cloud_list) : 
            self.cloud_array = np.vstack(self.cloud_list) # turns cloud list to an numpy array, long long list of all teh coordinates inside of cloud_list

            # print("Cloud array shape: ", np.shape(self.cloud_array))

            disx = self.cloud_array[:,0] - self.latest_pos.x
            disy = self.cloud_array[:,1] - self.latest_pos.y
            disz = self.cloud_array[:,2] - self.latest_pos.z


            xyzdist = np.column_stack((disx,disy,disz))
            # print("shape of x_and_y: ", np.shape((x_and_y)),"\n")
            # print("x_and_y: ", x_and_y, '\n') 

            norms = np.linalg.norm(xyzdist, axis = 1) # find norm given drone relative distance of x and y concatenated,
            # print("shape of norm: ", np.shape(norms),"\n")
            # print("Norm: ", norms, '\n') 
            bounded = self.cloud_array[norms < self.radius] #boolean mask to save bounded cloud points
            # print("BOUNDED: ", bounded)
            # print("cloud_array size: ", np.shape(self.cloud_array), '\n')
            # print("BOUNDED size: ", np.shape(bounded))

            #Make new bounded down-sampled cloud w/ voxel s, tuff
            self.voxel_array = self.voxel_size * np.unique(np.round( bounded / self.voxel_size), axis = 0) #voxel array of unique, occupied voxel spots 

            print("Voxel array 1ST TEN COOR: ", self.voxel_array[0:10,:])
            print("bounded array 1ST TEN COOR: ", bounded[0:10,:], "\n")

            return 1


      


    def run(self):
        while(not rospy.is_shutdown()): #TODO add smt when do FSM
            with self.lock:
                if self.is_odom == True:
                    # print("NOW HERE")
                    odom = self.latest_pos
                    #TODO add make_cloud here
                    if self.make_cloud(odom):
                        self.publ()
def main():
    rospy.init_node("Accumulator") #Make the node

    rate = rospy.Rate(10)

    Accumulator().run()

    rospy.spin()

if __name__ =="__main__":
    main()