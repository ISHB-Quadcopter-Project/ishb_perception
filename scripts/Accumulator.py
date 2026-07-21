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
        self.voxel_size = 0.067 #Width of voxels
        self.numframes = 10 # how long temporal window lasts in seconds * 10, how many saved clouds
        self.cloud_list = deque(maxlen = self.numframes)
        self.radius = 20 #Bounding radius
        self.cloud_array = None  
        self.voxel_array = None
        self.bounded = None

        #Cum Cloud
        self.cum_cloud = None

        #Timer
        rospy.Timer(rospy.Duration(.1), self.on_timer)  # 10 Hz

    def publ(self):
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        self.cum_cloud = self.make_pointcloud2_xyz32(header, self.voxel_array)
        self.pub.publish(self.cum_cloud)

    @staticmethod
    def make_pointcloud2_xyz32(header, points):
        """Build an XYZ32 PointCloud2 via one tobytes() instead of a Python
        per-point pack loop (which is what create_cloud_xyz32 does internally)."""
        points = np.ascontiguousarray(points, dtype=np.float32)  # (N,3), row-major x,y,z
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * points.shape[0]
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

        
    
    def cloud_to_xyz(self, msg): #TODO comment
        # For FAST-LIO's /cloud_registered: x,y,z are float32 at offsets 0,4,8
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('_pad', np.uint8, msg.point_step - 12)
        ])
        arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
        return np.column_stack((arr['x'], arr['y'], arr['z']))

    def odom_cb(self, msg):
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            #print("INSIDE HERE IS P: ", self.latest_pos)
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        self.latest_cloud = self.cloud_to_xyz(msg) 

        with self.lock:
            if len(self.latest_cloud):
                self.cloud_list.append(self.down_cloud(self.latest_cloud))
        # # print("one cloud:  ", self.latest_points, "\n")
        # print("sh of one cloud: ", np.shape(self.latest_cloud))
        # # print("sh of asdfasdfcloud: ", np.shape(self.latest_points[0,:]))
        # if len(self.latest_cloud) : self.cloud_list.append(self.latest_cloud) # this adds to cloud_list, the deque, list of n x 2 rows/coords 
            
    def down_cloud(self, cloud):
        #This get's the bin num, for assigning bin indexes to each downsampled voxel box
        bins = np.floor( cloud / self.voxel_size).astype(np.int64) 

        #This turns all the bin num to pos, by sub most neg xyz pt in cloud (found w/ axis = 0)
        pos_bins = bins - bins.min(axis=0)

        #Key is a 64 bit int, it all cloud pts in order: x, y, z. Uses pos bin so no need for 2s cmp or any goofy stuff
        key = (pos_bins[:, 0] << 42) | (pos_bins[:,1] << 21) | pos_bins[:, 2] 

        #Only grabbing indices of the unique pos x,y,z in 1D key (np.unique beta w/). Note these alr been voxeled
        _, first = np.unique(key, return_index = True) 

        #Now we have the indices of the unique voxel bins, we now use the og array of voxel bins, scaling by voxel size. 
        voxel_cen = (bins[first] + 0.5) * self.voxel_size #Adding +0.5 to move from corner voxel box to center

        return voxel_cen

    def make_cloud(self, odom):
        # print("DIST ODOM: ", odom, "\n")
        Odomx = odom.x
        Odomy = odom.y
        Odomz = odom.z
        self.cloud_array = np.vstack(self.cloud_list) #Turns the voxeled cloud into a numpy array of many voxeled clouds

        # print("Cloud array shape: ", np.shape(self.cloud_array))

        disx = self.cloud_array[:,0] - self.latest_pos.x
        disy = self.cloud_array[:,1] - self.latest_pos.y
        disz = self.cloud_array[:,2] - self.latest_pos.z


        xyzdist = np.column_stack((disx,disy,disz))
        # print("shape of xyzdist: ", np.shape((xyzdist)),"\n")
        # print("xyzdist: ", xyzdist, '\n') 

        norms = np.linalg.norm(xyzdist, axis = 1) # find norm given drone relative distance of x and y concatenated,
        # print("shape of norm: ", np.shape(norms),"\n")
        # print("Norm: ", norms, '\n') 
        self.bounded = self.cloud_array[norms < self.radius] #boolean mask to save bounded cloud points
        # print("BOUNDED: ", self.bounded)
        # print("cloud_array size: ", np.shape(self.cloud_array), '\n')
        # print("BOUNDED size: ", np.shape(self.bounded))

        self.voxel_array = self.down_cloud(self.bounded)

        # print("Voxel array 1ST TEN COOR: ", self.voxel_array[0:10,:])
        # print("bounded array shape: ", self.bounded.shape, "\n")
        # print("odommy:", self.latest_pos, "\n")
        if len(self.voxel_array):
            return 1
        return 0

    def run(self):
        rospy.spin()


    # def run(self):
    #     rate = rospy.Rate(2)
    #     while(not rospy.is_shutdown()): #TODO add smt when do FSM
    #         if self.is_odom == True:
    #             # print("NOW HERE")
    #             odom = self.latest_pos
    #             #TODO add make_cloud here
    #             if self.make_cloud(odom):
    #                 self.publ()
    #         rate.sleep()

    def on_timer(self,event):
        odom = self.latest_pos
        if self.is_odom and self.make_cloud(self.latest_pos):
            self.publ()

def main():
    rospy.init_node("Accumulator") #Make the node

    Accumulator().run()

    # rospy.spin()

if __name__ =="__main__":
    main()