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
        #because you are reinterpreting numpy arrays, point_cloud2 namespace message fields use raw bytes for information with a very specific formation, and 
        #msg.data = points.tobytes() is used, turning it into a contiguous array 
        #protects the information and dfines the type of each array value before turning it back
        points = np.ascontiguousarray(points, dtype=np.float32)  # (N,3), row-major x,y,z
        msg = PointCloud2()
        msg.header = header
        msg.height = 1 #height 1, being an unorganized point cloud
        msg.width = points.shape[0] # width is just how many points there are 
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False #little endian
        msg.point_step = 12
        msg.row_step = 12 * points.shape[0]
        msg.is_dense = True #Assume no NaN
        msg.data = points.tobytes()
        return msg

        
    
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

    def odom_cb(self, msg):
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            #print("INSIDE HERE IS P: ", self.latest_pos)
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        self.latest_cloud = self.cloud_to_xyz(msg) 

        reduced = self.down_cloud(self.latest_cloud) #Downsample outside lock to avoid thread being locked excessively

        with self.lock:
            if len(self.latest_cloud):
                self.cloud_list.append(reduced)
        
    def down_cloud(self, cloud):
        #This get's the bin num, for assigning bin indexes to each downsampled voxel box
        bins = np.floor( cloud / self.voxel_size).astype(np.int64) 

        #This turns all the bin num to pos, by sub most neg xyz pt in cloud (found w/ axis = 0)
        pos_bins = bins - bins.min(axis=0)

        #Key is a 64 bit int, it all cloud pts in order: x, y, z. Uses pos bin so no need for 2s cmp or any goofy stuff
        key = (pos_bins[:, 0] << 42) | (pos_bins[:,1] << 21) | pos_bins[:, 2] 

        #Only grabbing indices of the unique pos x,y,z in 1D key (np.unique beta w/). Note these alr been voxeled
        _, first = np.unique(key, return_index = True) 

        #Now we have the indices of the unique voxel bins b/c first collerates to rows, we now use the og array of voxel bins, scaling by voxel size. 
        voxel_cen = (bins[first] + 0.5) * self.voxel_size #Adding +0.5 to move from corner voxel box to center

        return voxel_cen

    def make_cloud(self, odom):
        #use latest saved odometry data
        Odomx = odom.x
        Odomy = odom.y
        Odomz = odom.z
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
        self.voxel_array = self.down_cloud(self.bounded)   # this is not below vstack, as its done before bounding to lessen how many thigns go into down_cloud
        #if voxel array, is not empty, return 1, else return 0
        #in other words, return 1 unless if no occupied voxels, return 0
        #saves us from publishing empty map, which would cause error at worst and waste memory at best
        if len(self.voxel_array):
            return 1
        return 0

    def run(self):
        rospy.spin()


    def on_timer(self,event):
        odom = self.latest_pos
        #if odometry and unique cloud window is open, then publish
        if self.is_odom and self.make_cloud(self.latest_pos):
            self.publ()

def main():
    rospy.init_node("Accumulator") #Make the node

    Accumulator().run()

    # rospy.spin()

if __name__ =="__main__":
    main()