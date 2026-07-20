#!/usr/bin/env python3
import rospy
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
        # self.latest_points = None

        #Intializing publisher
        self.pub = rospy.Publisher("/newCloud", PointCloud2, queue_size = 10)

        #Intializing subscriber
        self.sub_o = rospy.Subscriber("/Odometry", Odometry, self.odom_cb, queue_size = 10)

        #Intializing subscriber
        self.sub_c = rospy.Subscriber("/cloud_registered", PointCloud2, self.cloud_cb, queue_size = 10)

        #Flag to see if there is available odom data
        self.is_odom = False

        #For make cloud
        self.voxel_num = 0.5

        self.cloud_list = deque(maxlen = 50)

    def publ(self):
        pass
        #TODO call the make_cloud func and publish

    def odom_cb(self, msg):
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            #print("INSIDE HERE IS P: ", self.latest_pos)
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        latest_points = np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y"), skip_nans = True)))
        print("one cloud:  ", latest_points, "\n")
        print("sh of one cloud: ", np.shape((latest_points)))
        print("sh of asdfasdfcloud: ", np.shape(latest_points[0,:]))
        if len(latest_points) : self.cloud_list.append(latest_points)
            

    def make_cloud(self, odom):
        # print("DIST ODOM: ", odom, "\n")
        Odomx = odom.x
        Odomy = odom.y
        Odomz = odom.z
        if len(self.cloud_list): print("Cloud list: ", np.shape(self.cloud_list[1]))

        array = self.cloud_list
        # print("len of cloud list: ", len(self.cloud_list))
        # print("SHAPE: ", np.shape(array))

        # norms = np.linalg.norm(array(:,1) - odom , axis = 1)


        #Make new bounded down-sampled cloud w/ voxel stuff

      


    def run(self):
        while(not rospy.is_shutdown()): #TODO add smt when do FSM
            with self.lock:
                if self.is_odom == True:
                    # print("NOW HERE")
                    odom = self.latest_pos
                    #TODO add make_cloud here
                    self.make_cloud(odom)
def main():
    rospy.init_node("Accumulator") #Make the node

    Accumulator().run()

    rospy.spin()

if __name__ =="__main__":
    main()