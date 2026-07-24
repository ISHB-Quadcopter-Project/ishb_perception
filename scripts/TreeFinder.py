#!/usr/bin/env python3
import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
import threading
import math
import numpy as np
import sklearn

#TODO We want to do two things A) find poss trees B) sep class at least, goes up to poss tree, and estimates radius, and reonsiders dist from tree for cam, and reconsiders tree placements
#b would happen before and while going towards tree, 
#When state 3 happens (assecnetion), that's when we know for certain the rad of the tree, and can accurately place waypt,next waypoint can b determined

class TreeFinder:
    def __init__(self): 
        self.lock = threading.Lock()

        #Publish beacon/waypoint for found trees
        self.pub = rospy.Publisher("/Tree_Cand", PointStamped, queue_size = 10)

        #Sub to cum cloud
        self.sub = rospy.Subscriber("/Cum_Cloud", PointCloud2, self.cloud_cb, queue_size = 10)

        #Timer
        # rospy.Timer(rospy.Duration(.1), self.on_timer)  # 10 Hz

        self.latest_cloud = None
        self.processed_cloud = None
        self.tree_cands = None
        self.z_cut_one = 4.3885
        self.z_cut_two = 10


    def run(self):
        rospy.spin()

    #TODO cb for getting cloud data, consider downsampling once more here, or doing 2d slices by callign func
    def cloud_cb(self, msg):
            self.latest_cloud = self.cloud_to_xyz(msg) 

            # with self.lock:
            print(self.z_cut_one,"HERE is cut for 1st z: ", self.cut_cloud(self.latest_cloud, self.z_cut_one))

    #TODO func to do 2d slice of cloud data, think optimizing, don't jsut do 1 slice, do 2-3 to know its a tree (ver. cyl ah shape ish)
    def cut_cloud(self, uncut_cloud, z_height):
        print("HERE is pre processed z: ", uncut_cloud[:,2])
        return uncut_cloud[uncut_cloud[:,2]== z_height]
         
        # bobby = []
        # # print("uncut cloud z col: ", uncut_cloud[:,2])
        # for vox_number in self.voxel_range(self.z_cut_one, 50):
        #     bobby.append(uncut_cloud[uncut_cloud[:,2]== vox_number.astype(float32)])
        #     print(vox_number)
        # return bobby

    # def voxel_range(self, lower, voxel_list_l):
    #     voxel_list = []
    #     j = 1
    #     for i in range(voxel_list_l):
    #         voxel_list.append(lower * 0.067 * j)
    #         j += 1
    #     # print(voxel_list)
    #     return voxel_list


    def cloud_to_xyz(self, msg):
        # For FAST-LIO's /cloud_registered: x,y,z are float32 at offsets 0,4,8

        #defining the type of the datanp.dyte('name of this field', 'type of field', 'how many bits this part takes')
        #here, four fields fields can be stacked to be define the fields of incoming PointCloud2 message
        dtype = np.dtype([     
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('_pad', np.uint8, msg.point_step - 12)
        ])
        #contiguous array not needed, as not turning numpy-> binary blob, its just binary blob parsed into binary blob
        #
        arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
        # the buffer is the the data being locked, and what threads write to, its a sort of temp memory(doesnt need to be locked, we have our own version of msg due to callback writing to self.something)
        # frombuffer reads data without copying it, treating it as a 1d array, and keeping only(in this case)
        # reads msg.data in the thread, interprets as .data, which is a Pointcloud2 message, dtype recreating our version of the Pointcloud2 message format, 
        # with the size of the buffer being read as the size of the whole pointcloud
        # (how many points there are in the pointcloud, given by msg.heigh/width, parameters of passed message)
        return np.column_stack((arr['x'], arr['y'], arr['z']))

    
         
         
    #TODO ontimer functin, reads possible trees, does beacon publishing

    #TODO func to cluster data and find poss trees

def main():
    rospy.init_node("TreeFinder") #Make the node

    TreeFinder().run()

if __name__ =="__main__":
    main()