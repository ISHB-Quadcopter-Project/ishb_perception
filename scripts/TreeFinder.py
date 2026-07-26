#!/usr/bin/env python3
import os
import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
import threading
import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque 

#TODO We want to do two things A) find poss trees B) sep class at least, goes up to poss tree, and estimates radius, and reonsiders dist from tree for cam, and reconsiders tree placements
#b would happen before and while going towards tree, 
#When state 3 happens (assecnetion), that's when we know for certain the rad of the tree, and can accurately place waypt,next waypoint can b determined

class TreeFinder:
    def __init__(self): 
        self.lock = threading.Lock()

        #Publish beacon/waypoint for found trees
        self.pub_beacon = rospy.Publisher("/Beacons", PointCloud2, queue_size = 10)

        self.pub_cloud = rospy.Publisher("/Z_CLUSTERS", PointCloud2, queue_size = 10)

        #Sub to cum cloud
        self.sub = rospy.Subscriber("/Cum_Cloud", PointCloud2, self.cloud_cb, queue_size = 10)
        self.cloud_section_one = None
        self.cloud_section_two = None

        #Timer
        rospy.Timer(rospy.Duration(0.1), self.on_timer)  # 10 Hz

        self.latest_cloud = None
        self.processed_cloud_low = None 
        self.processed_cloud_high = None
        self.tree_cands = None

        # self.z_low_one = round(5 * 0.067, 5)
        # self.z_high_one = round(6 * 0.067, 5)

        # self.z_low_two = round(15 * 0.067, 5)
        # self.z_high_two = round(20 * 0.067, 5)
        
        self.pancake_stacks = 5
        self.pancake_start = round(5 * 0.067, 5)
        self.pancake_gap = round(4 * 0.067, 5)
        self.pancake_thickness = round(1 * 0.067, 5)

        self.xy = None
        self.centroid_list = None

        self.eps         = rospy.get_param("~dbscan_eps", 0.25)
        self.min_samples = rospy.get_param("~dbscan_min_samples", 5)

        self.debug_plot  = rospy.get_param("~debug_plot", True)
        self.plot_period = rospy.get_param("~plot_period", 2.0)   # seconds
        self.plot_dir    = os.path.expanduser(
            rospy.get_param("~plot_dir", "~/ishb_ws/debug_plots"))

        if self.debug_plot:
            os.makedirs(self.plot_dir, exist_ok=True)
            self._fig, self._ax = plt.subplots(figsize=(6, 6), dpi=90)
            self._last_plot_tall = rospy.Time(0)
            self._last_plot_short = rospy.Time(0)

        self.cand_trees = np.zeros(self.pancake_stacks,dtype = object)
        self.last_plot = None
        self.processed_cloud_list = []

        self.publish_list = []

        


    def run(self):
        rospy.spin()

    

    def cloud_cb(self, msg):
        self.latest_cloud = self.cloud_to_xyz(msg)
        self.processed_cloud_list.clear()

        for i in range(self.pancake_stacks):
            mid_height = self.pancake_start + (self.pancake_gap * i) 
            processed_cloud = self.cut_cloud(self.latest_cloud, mid_height)
            self.processed_cloud_list.append(processed_cloud)
            # print("HERE is pro clouod list: ", self.processed_cloud_list)
            # print("HERE is pro clouod list len: ", len(self.processed_cloud_list))
            
            # # with self.lock:
            # self.processed_cloud_low = self.cut_cloud(self.latest_cloud, self.z_low_one, self.z_high_one) 
            # self.processed_cloud_list.append(self.processed_cloud_low)
            # #TODO think about being able to iterate through more voxel heights

            # self.processed_cloud_high = self.cut_cloud(self.latest_cloud, self.z_low_two, self.z_high_two)
            # self.processed_cloud_list.append(self.processed_cloud_high)
            # # print("HERE is cut for 1st z: ", self.processed_cloud)

    def cut_cloud(self, uncut_cloud, z_mid):
        # print("HERE is pre processed z: ", uncut_cloud[:,2])
        z_high = z_mid + self.pancake_thickness/2
        z_low = z_mid - self.pancake_thickness/2
        #This mask looking from points in a range of z values. However, these z's step by the voxel_size naturally (cloud alr voxeled)
        mask = (uncut_cloud[:,2] >= z_low) & (uncut_cloud[:,2] <= z_high)

        #Apply mask to the uncut_cloud, to "cut" it at our range of z values
        cut_cloud = uncut_cloud[mask]
        intcast_cloud = (cut_cloud*1000000).astype(np.int64) 

        #Key is a 64 bit int, it all cloud pts : x, y ONLY. 
        key = (intcast_cloud[:, 0] << 21) | (intcast_cloud[:,1]) 

        #Only grabbing indices of the unique pos x,y in 1D key (np.unique beta w/).
        _, first = np.unique(key, return_index = True) 

        #Below is for rviz publishing
        #Return cut_cloud with indices that only include uniqe x,y's
        # print("HERE is shape not_includez: : ", np.shape(cut_cloud[first,0:2]))
        # self.publish_list.append(cut_cloud[first]) #TODO publish stuff

        

        # print("publish list size: asdfasdfasdfasdf",np.shape(self.publish_list))
        # print("cut cloud shape asdfasfasfasdfasdf " , np.shape(cut_cloud[first])

        #TODO maybe here, and not in callback or in self, incopeerate dict iterating dif voxel height???
        # if(z_low == self.z_low_one):
        #     self.cloud_section_one =cut_cloud[first]
        # else:
        #     self.cloud_section_two =cut_cloud[first]


        return cut_cloud[first,0:2] 

    def on_timer(self,event):
        with self.lock:
            if len(self.processed_cloud_list):
                for pancake_num in range(self.pancake_stacks):
                    e_array = self.centroid_finder(pancake_num) #TODO pass in which and for loop maybe here, passing in "which", the index of processed_cloud_list
                    self.cand_trees[pancake_num] = e_array
            # print("HERE is self.cand_tree: ", self.cand_trees)
            print("HERE is self.cand_tree shape: ", self.cand_trees.shape)

            # self.publ() #TODO publilsh stuff
        #TODO  call cluster categorizing steps 2-4 funcs here
            

        #TODO artifact, and commented out pub in cu_cloud update this beta
        # if len(self.processed_cloud_low): #and len(self.processed_cloud_high):
        #     # self.clustering(self.processed_cloud_low, 0)
        #     #self.clustering(self.processed_cloud_high, 1)
        #     self.publ()    

    def centroid_finder(self, which): #TODO pass in which here, and incoperate logic for it
        # if len(self.processed_cloud_low): #and len(self.processed_cloud_high):
        #             labels, n_clusters = self.clustering(self.processed_cloud_low, 0)
        #             # print("labels shape ::::::::::",labels.shape)
        #             # print("xy ::::::::::",self.xy.shape)
        #             # print("labels == 3::::::::::",(labels == 3).shape)

        #             #self.clustering(self.processed_cloud_high, 1) 

        # print("HERE is which inside of centriod findeer: ", which)
        # print("proc cloud list length::::::::::::: " ,len(self.processed_cloud_list))
        if len(self.processed_cloud_list) > which:
            processed_cloud = self.processed_cloud_list[which]
            if len(processed_cloud):
                labels, n_clusters = self.clustering(processed_cloud, which)

            self.centroid_list = np.zeros((n_clusters-1,3))

            e_array = np.zeros((n_clusters-1, 6))#TODO np zeros?

            for clustnum in range(n_clusters-1):
                curr_clust = self.xy[labels == clustnum]
                xmean = np.mean(curr_clust[:,0])
                ymean = np.mean(curr_clust[:,1])
                self.centroid_list[clustnum,0] = xmean
                self.centroid_list[clustnum,1] = ymean
                self.centroid_list[clustnum,2] = 1.67
                # print("HERE is self.centriod_list: ", self.centroid_list)

                #TODO call eigen, pass in curr_clust and xmean and ymean
                hor_rms, ver_rms, elongation_num = self.eigen(curr_clust, xmean, ymean)
                e_array[clustnum] = [hor_rms, ver_rms, elongation_num, clustnum,xmean,ymean]

                # np.append(e_array, np.array([hor_rms, ver_rms, elongation_num, clustnum,xmean,ymean]))
                # e_array.append(np.array([hor_rms, ver_rms, elongation_num, clustnum,xmean,ymean]),axis = 0)

            return e_array

    def eigen(self, curr_clust, xmean, ymean):
        normalized = curr_clust - np.array([xmean,ymean]) #To do it at origin
        #  print("HERE is normalized shape: ", normalized.shape)

        cov_matrix = np.cov(normalized, rowvar = False)
        #  print("HERE is cov matrix: ", cov_matrix)
        #  print("HERE is cov matrix shape: ", cov_matrix.shape)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # print("Eigenvalues:\n", eigenvalues)
        # print("Eigenvectors:\n", eigenvectors)
        hor_rms = math.sqrt(eigenvalues[0])
        ver_rms = math.sqrt(eigenvalues[1])

        elongation_num = math.sqrt(hor_rms/ver_rms)

        print("HERE is hor_rms: ", hor_rms)
        print("HERE is ver_rms: ", ver_rms)
        print("HERE is elong: ", elongation_num)

        return hor_rms, ver_rms, elongation_num  

    def clustering(self, points, which): 
        
        """points: (N,2) or (N,3) float array, metres.
        Returns (labels, n_clusters). Label -1 == noise."""
        if points.shape[0] < self.min_samples: #num of coordinates to cluster < min samples
            return np.full(points.shape[0], -1, dtype=int), 0 #return [-1,-1,-1] labels anda zero b/s not eenoguh pts to even make one cluster (def no trees nearby)

        #creates pointer to the points array, as long as the type is dtype 
        #more efficient than np.array which makes new nparray object, this is like a conditional to make sure of the type, and an C (call by ref) array
        self.xy = np.asarray(points[:, :2], dtype=np.float64) 
        

        #creates scanning object, db
        #epsilon = neighborhood radius param
        #min samples/points per cluster 
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples) 

        #returns numpy 1D array of labels(int numwhich cluster each point belongs) aligned with the rows of xy asarray
        labels = db.fit_predict(self.xy)

        #Set on labels to get the unique labels of the label array (that's for every pt)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #subtracts 1 if outlier
        n_noise = int(np.count_nonzero(labels == -1)) #number of outliers

        rospy.loginfo_throttle(
            1.0, "DBSCAN: %d pts -> %d clusters, %d noise",
            self.xy.shape[0], n_clusters, n_noise)

        if self.debug_plot:
            #TODO EACH new dict will be on their own timer also, if we want to plot seperately, since only one "whcih" layer would run at a time. 
            #coudl also think about changing the savecluterplot funciton to have multiple plots

            now = rospy.Time.now()
            self._save_cluster_plot(self.xy, labels, n_clusters, now ,which)
            # if ( self._last_plot[which] == None):
            #     self._last_plot[which].append(now)
            #     self._save_cluster_plot(self.xy, labels, n_clusters, now ,which)

            # elif (now - self._last_plot[which]).to_sec() >= self.plot_period: #This ensure one short plot is created every plot period
            #     self._last_plot[which] = now
            #     self._save_cluster_plot(self.xy, labels, n_clusters, now ,which)
            # if which:
            #     now = rospy.Time.now()
            #     if (now - self._last_plot_short).to_sec() >= self.plot_period: #This ensure one short plot is created every plot period
            #         self._last_plot_short = now
            #         self._save_cluster_plot(self.xy, labels, n_clusters, now_short ,which)
            # else:
            #     now = rospy.Time.now()
            #     if (now - self._last_plot_tall).to_sec() >= self.plot_period: #This ensure one tall plot is created every plot period
            #         self._last_plot_tall = now
            #         self._save_cluster_plot(self.xy, labels, n_clusters, now_tall ,which)                    

        return labels, n_clusters

    def _save_cluster_plot(self, xy, labels, n_clusters, stamp ,which): #TODO update the which swiching logic, , mayb ea folder for each time step, chat will do though
        ax = self._ax
        ax.cla()

        noise = labels == -1
        if noise.any():
            ax.scatter(xy[noise, 0], xy[noise, 1],
                       s=2, c="0.75", marker=".", linewidths=0, label="noise")

        ids = np.unique(labels[~noise])
        if ids.size:
            colors = plt.cm.Spectral(np.linspace(0.0, 1.0, ids.size))
            for k, col in zip(ids, colors):
                m = labels == k
                ax.scatter(xy[m, 0], xy[m, 1], s=6, color=col, linewidths=0)
                # label at centroid instead of a legend entry per tree
                ax.annotate(str(k), (xy[m, 0].mean(), xy[m, 1].mean()),
                            fontsize=7, color="k",
                            ha="center", va="center")
            

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        z = self.pancake_start + (self.pancake_gap * which) 

        ax.set_title("DBSCAN eps=%.2f min_samples=%d - %d clusters, Z = %.2fm"
                                 % (self.eps, self.min_samples, n_clusters,z))
        path = os.path.join(self.plot_dir, "%.2f_tall_clusters_%.2f.png" % (z , stamp.to_sec()))
        self._fig.savefig(path, bbox_inches="tight")


        rospy.loginfo("wrote %s", path)
        
        ax.grid(True, linewidth=0.3, alpha=0.5)
    


    #-------Helper funcs and pub------

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

    def make_pointcloud2_xyz32(self, header, points):
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
            msg.is_bigendian = False
            msg.point_step = 12
            msg.row_step = 12 * points.shape[0]
            msg.is_dense = True
            msg.data = points.tobytes()
            return msg     

    def publ(self):
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())

        cluster_cloud = self.make_pointcloud2_xyz32(header, np.array(self.publish_list))

        beacons = self.make_pointcloud2_xyz32(header, self.centroid_list)


        self.pub_cloud.publish(cluster_cloud)
        self.publish_list = []

        self.pub_beacon.publish(beacons)
    

def main():
    rospy.init_node("TreeFinder") #Make the node

    TreeFinder().run()

if __name__ =="__main__":
    main()