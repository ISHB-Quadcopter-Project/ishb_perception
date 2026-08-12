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
from collections import deque, defaultdict
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from common import *

from geometry_msgs.msg import PoseStamped



#TODO We want to do two things A) find poss trees B) sep class at least, goes up to poss tree, and estimates radius, and reonsiders dist from tree for cam, and reconsiders tree placements
#b would happen before and while going towards tree, 
#When state 3 happens (assecnetion), that's when we know for certain the rad of the tree, and can accurately place waypt,next waypoint can b determined

class TreeFinder:
    """!@brief Still in development!"""
    def __init__(self): 
        self.lock = threading.Lock()

        #Publish beacon/waypoint for found trees
        self.pub_beacon = rospy.Publisher("/Beacons", PointCloud2, queue_size = 10)

        self.pub_cloud = rospy.Publisher("/Z_CLUSTERS", PointCloud2, queue_size = 10)

        self.pub_line = rospy.Publisher("LINES", PointCloud2, queue_size = 10)

        self.pub_hespline = rospy.Publisher("HESP_LINES", PointCloud2, queue_size = 10)

        self.pub_not_line = rospy.Publisher("NOT_LINES", PointCloud2, queue_size = 10)

        self.pub_persisted = rospy.Publisher("PERSISTED", PointCloud2, queue_size = 10)

        self.pub_to_go_pt = rospy.Publisher("TOGO", PointStamped, queue_size = 10)

        self.pub_text = rospy.Publisher('/TEXT', MarkerArray, queue_size=10)

        #Sub to cum cloud
        self.sub = rospy.Subscriber("/Cum_Cloud", PointCloud2, self.cloud_cb, queue_size = 10)
        self.cloud_section_one = None
        self.cloud_section_two = None

        #Timer
        self.on_timer_dur = 0.1
        rospy.Timer(rospy.Duration(self.on_timer_dur), self.on_timer)  # 10 Hz

        self.persistence_dur = 3
        rospy.Timer(rospy.Duration(self.persistence_dur), self.persistence_timer)  # 1 Hz

        self.latest_cloud = None
        
        self.pancake_stacks = 7
        self.pancake_start = round(5 * 0.067, 5)  #5 #TODO make some sorta global arg from config?, or maybe a func that reads odom and updates it
        self.pancake_gap = round(1* 0.067, 5) #TODO This has to be small for new alg
        self.pancake_thickness = round(3 * 0.067, 5)
        self.mid_height = round(self.pancake_stacks/2) *self.pancake_gap + self.pancake_start #Not including in calcualtion b/c so small
        self.xy = None
        self.centroid_list = None

        self.eps         = rospy.get_param("~dbscan_eps", 0.25)
        self.min_samples = rospy.get_param("~dbscan_min_samples", 5)

        self.debug_plot  = rospy.get_param("~debug_plot", False)
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
        self.processed_cloud_list = deque(maxlen = self.pancake_stacks*2)

        self.publish_list = deque(maxlen =self.pancake_stacks*1)

        #------New alg---------
        self.mid_z_dict = defaultdict(dict) #Stores relevant info for mid z slice clusters: cluster, num pts in cluster, centriod
        self.mid_n_clusters = 0
        self.clustered_cloud_list = deque(maxlen = 50)
        self.all_vpancakes = None
        self.pub_line_list= []
        self.pub_not_line_list= []
        self.leaf_size = 80

        self.text_dict = defaultdict(dict)
        self.mid_count = 0
        self.angle_threshold = 25 #degrees, for determining if a cluster is vertical or not

        self.kd_tree_PCA_done = False

        self.persistence_list = []
        self.tol = 0.08
        self.goal_tol = 1

        self.persistence_freq = 0

        self.pub_persisted_array = np.zeros(0)

        #Maximum possible counts is the duration of persistence, divided by how often you add to the persistence_list
        self.max_pers_counts = self.persistence_dur / self.on_timer_dur
        self.persisted_scores_weight = 1
        self.norms_scores_weight = 1

        self.all_persisted_array = np.zeros(0)
        self.qbeen = np.zeros(0)

        self.linelen = 2
        self.hlines = []

        self.freq_percent = 0.5

        self.trunc_deci = 5
        self.trunc_factor = 10 ** 5

        self.all_p1 = np.zeros(0)

        #TODO organize init, and add a section of all things can tune

        # self.zzpoints = np.array([
        #     [0, 0], [12, 19], [24, 0], [36, 19], [48, 0], [60, 19], [72, 0],
        #     [2, 3.167], [4, 6.333], [6, 9.5], [8, 12.667], [10, 15.833],
        #     [14, 15.833], [16, 12.667], [18, 9.5], [20, 6.333], [22, 3.167],
        #     [26, 3.167], [28, 6.333], [30, 9.5], [32, 12.667], [34, 15.833],
        #     [38, 15.833], [40, 12.667], [42, 9.5], [44, 6.333], [46, 3.167],
        #     [50, 3.167], [52, 6.333], [54, 9.5], [56, 12.667], [58, 15.833],
        #     [62, 15.833], [64, 12.667], [66, 9.5], [68, 6.333], [70, 3.167],
        # ], dtype=np.float32)

        #Odom var to hold the x,y,z odom data
        self.latest_pos = None

        self.sub = rospy.Subscriber("/Odometry", Odometry, self.odom_cb, queue_size = 10)
        
        #Flag to see if there is available odom data to check dist_to_goal
        self.is_odom = False

        # Topic super takes
        self.pub_super = rospy.Publisher("/super/goal", PoseStamped, queue_size = 10) 

        



        #TODO OLD STUFF: Tree Confirmation Parameters

        #Step 1: Is Line
        self.hor_rms_threshold = 0.2 #Measure of how spread horizontally
        self.elongation_num_threshold = 1 #ranges 0-1, higher is more "circular"
        


    def run(self):
        rospy.spin()

    
#---------------------------------------------------------------------------------------Subscriber Thread-----------------------------------------------------------------------------------------
    def odom_cb(self, msg):
        """!@brief Callback function for the /Odometry topic
            @details Updates the latest odometry position and sets the is_odom flag to True. Odom data is stored in a list for the odom_watchdog to check if the drone is moving.
            @note self.lock is used to ensure other areas of code using odom data don't get partial data, as this callback is in a separate thread
            @param msg The Odometry message received from the /Odometry topic"""
        
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        """!@brief Callback function for the /Cum_Cloud topic.
            @details Adds clouds with specified z-ranges using cut_cloud to a list for centroid_finder
            @param msg The PointCloud2 message received from the /Cum_Cloud
            @see cut_cloud"""
        self.latest_cloud = cloud_to_xyz(msg)

        with self.lock:
            for i in range(self.pancake_stacks):
                cur_mid_height = self.pancake_start + (self.pancake_gap * i)  #Middle height of cur pancake looking at
                processed_cloud = self.cut_cloud(self.latest_cloud, cur_mid_height)
                self.processed_cloud_list.append(processed_cloud)

    def cut_cloud(self, uncut_cloud, z_mid):
        """!@brief Cuts a point cloud to a specified z-range and returns unique x,y coordinates inside range.
            @details A boolean for the specified z-range is created, to "cut" the cloud. A bit-packed key is created for the x,y coordinates, to find make finding unique x,y coordinates faster. This part is similar to Accumulator.Accumulator.down_cloud
            @param uncut_cloud The numpy array point cloud to cut
            @param z_mid The middle height of the z-range to cut
            @return A numpy array of shape (N, 2) containing the x,y coordinates of the points in the specified z-range"""
        # print("HERE is pre processed z: ", uncut_cloud[:,2])
        z_high = z_mid + self.pancake_thickness/2
        z_low = z_mid - self.pancake_thickness/2

        #This mask looking from points in a z slice. However, these z's step by the voxel_size naturally (cloud alr voxeled)
        mask = (uncut_cloud[:,2] >= z_low) & (uncut_cloud[:,2] <= z_high)

        #Apply mask to the uncut_cloud, to "cut" it at our z slice
        cut_cloud = uncut_cloud[mask]
        intcast_cloud = (cut_cloud*1000000).astype(np.int64) 

        #Key is a 64 bit int, it all cloud pts : x, y ONLY. 
        key = (intcast_cloud[:, 0] << 21) | (intcast_cloud[:,1]) 

        #Only grabbing indices of the unique pos x,y in 1D key (np.unique beta w/).
        _, first = np.unique(key, return_index = True) 

        #Below is for rviz publishing
        # self.publish_list = np.append(self.publish_list,cut_cloud[first],axis = 0)
        self.publish_list.append(cut_cloud[first])

        # print("publish list size: asdfasdfasdfasdf",np.shape(self.publish_list))
        # print("cut cloud shape asdfasfasfasdfasdf " , np.shape(cut_cloud[first]))

        #Return cut_cloud with indices that only include uniqe x,y's
        return cut_cloud[first,0:2] 


#--------------------------------------------------------------------------on_timer Thread (that sub and pub both depend on)-------------------------------------------------------------------------------
    def on_timer(self,event):
        """!@brief Timer callback to call centroid_finder on each z-slices. Calls publ too.
            @details This will pass in a bool flag to centorid_finder dictating whether it is the mid z-slice. If kd_tree_PCA is done, based on a flag, then relevant list and dicts for this class's operations are cleared to ensure data is refreshed.
            @param event An object of TimerEvent, automatically created every time rospy.Timer fires.
            @see centroid_finder"""
        i = 1

        self.mid_count = 0 #reset the mid_count for kd_tree_PCA
        with self.lock:
            if len(self.processed_cloud_list):
                half = self.pancake_stacks / 2
                mid_num = math.floor(half + 0.5)
                for pancake_num in range(self.pancake_stacks):
                    is_mid = False
                    # print("HERE is mid_num: ", mid_num)

                    if i == mid_num: #Checking if at the mid z-slice
                        is_mid = True

                    self.centroid_finder(pancake_num, is_mid)

                    # if e_array is not None:
                    #     # not_zero_mask = [e_array != (0,0,0,0,0,0)]
                    #     # print("EARRAY: ", e_array)
                    #     # print("EARRAY shape: ", e_array.shape)

                    #     not_zero_mask = np.any(e_array != 0, axis =1)
                    #     # print("HERE IS not zero mask: ", not_zero_mask)
                    #     cleaned_e_array = e_array[not_zero_mask]

                    #     self.cand_trees[pancake_num] = cleaned_e_array

                    i += 1

                    
                        

                            #TODO somehwere in on_timer, need to call func for kd tree and PCA
                # print("HERE is self.cand_tree: ", self.cand_trees)
                # print("HERE is self.cand_tree shape: ", self.cand_trees.shape
                if self.kd_tree_PCA_done == True:
                    self.publ()
                    self.pub_line_list.clear()
                    self.pub_not_line_list.clear()
                    self.text_dict.clear()
                    self.mid_z_dict.clear() #Clear dict for mid slice, before poulate again with new clustering
                    self.clustered_cloud_list.clear()
            #TODO  call cluster categorizing steps 2-4 funcs here

    def centroid_finder(self, which, is_mid):
        """!@brief Calls clustering on a specified z-slice. If it is the middle z-slice and not line shaped, then saves relevant info for kd_tree_PCA and also publishing text.
            @see is_line @see clustering @see kd_tree_PCA
            @param which An integer value representing which pancake is to be clustered.
            @todo return is artifact of e_array stuff"""

        # print("HERE is which inside of centriod findeer: ", which)
        # print("proc cloud list[0]: " ,self.processed_cloud_list[0])
        self.kd_tree_PCA_done = False
        if len(self.processed_cloud_list) > which:
            processed_cloud = self.processed_cloud_list[which]
            if len(processed_cloud):
                labels, n_clusters = self.clustering(processed_cloud, which)

                # self.centroid_list = np.zeros((n_clusters-1,3))
                if n_clusters > 0:
                    # e_array = np.zeros((n_clusters-1, 6))
                    self.clustered_cloud_list.append(self.xy[labels != -1])
                    # print("HERE is clustered_cloud_list: ", self.clustered_cloud_list)

                    for clustnum in range(n_clusters-1):
                        curr_clust = self.xy[labels == clustnum]
                        # print("\n------HERE is curr_clust: ", curr_clust, "\n")
                        xmean = np.mean(curr_clust[:,0])
                        ymean = np.mean(curr_clust[:,1])

                        #Getting relevant cluster info from eigen func
                        hor_rms, ver_rms, elongation_num = self.eigen(curr_clust, xmean, ymean)

                        clust_name = f"Cluster {self.mid_count}"

                        #Populate mid_z_dict if at mid z-sclie
                        if is_mid and not self.is_line(hor_rms, elongation_num):
                            # print("HERE is curr_clsut: ", curr_clust)
                            num_pts = curr_clust.shape[0]

                            if num_pts < 1000: #Checking if the cluster is absurd

                                #TODO replace this call with the filtering func
                                # print("HERE is num of ptS: ", num_pts)
                                self.mid_z_dict[clust_name]["num_pts"] = num_pts
                                self.mid_z_dict[clust_name]["xmean"] = xmean
                                self.mid_z_dict[clust_name]["ymean"] = ymean

                                #Putting hor_rms, xmean, and ymean in a dict dynamically to pub text
                                self.text_dict[clust_name]["hor_rms"] = hor_rms
                                self.text_dict[clust_name]["ver_rms"] = ver_rms
                                self.text_dict[clust_name]["xmean"] = xmean
                                self.text_dict[clust_name]["ymean"] = ymean
                                self.text_dict[clust_name]["elongation_num"] = elongation_num

                                self.mid_count += 1
                                # print("HERE is mid_count: ", self.mid_count)
                        



                            # print("HERE is mid_z_dict: ", self.mid_z_dict)

                                #Populating alr instantiated numpy array in mem. This array holds cluster info for all clusters in a z "pancake" slice
                                # e_array[clustnum] = [hor_rms, ver_rms, elongation_num, which,xmean,ymean]

                            # print("HER is e_array shape: ", e_array.shape)
                        #TODO if else statement for returned value for is big func ARTIFACT???

                    #TODO
                    # print("HERE is count: ", count)

                    if which == self.pancake_stacks - 1: 
                        # print("before calling kd_tree_PCA")
                        self.kd_tree_PCA(self.mid_count) #Call #TODO filtering func, after for loop so dict is fully populated
                        # print("after calling kd_tree_PCA")
                    # return e_array
                return None

    #---Fork 1 called by centriod_finder---
    def clustering(self, points, which): 
        """!@brief Clusters a point cloud using DBSCAN and returns the labels and number of clusters.
            @details calls _save_cluster_plot for debugging purposes.
            @return The labels and number of clusters"""
        
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

        return labels, n_clusters

    def _save_cluster_plot(self, xy, labels, n_clusters, stamp ,which): #TODO update the which swiching logic, , mayb ea folder for each time step, chat will do though
        """!@brief Saves a plot of the clustered point cloud for debugging purposes."""
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
    
    #---Fork 2 called by centriod_finder---
    def eigen(self, curr_clust, xmean, ymean):
            """!@brief Calculates the eigenvalues and eigenvectors of a cluster from the covariance matrix.
                @return The horizontal RMS, vertical RMS, and elongation number of the cluster."""
            # print("\n------INSIDE EIGEN------")
            # print("xmean: ", xmean)
            # print("ymean: ", ymean, "\n")
            if curr_clust.size != 0:
            
                normalized = curr_clust - np.array([xmean,ymean]) #To do it at origin
                # print("HERE is normalized shape: ", normalized.shape)
        
                cov_matrix = np.cov(normalized, rowvar = False)
                #  print("HERE is cov matrix: ", cov_matrix)
                #  print("HERE is cov matrix shape: ", cov_matrix.shape)
        
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                # print("Eigenvalues:\n", eigenvalues)
                # print("Eigenvectors:\n", eigenvectors)
                hor_rms = math.sqrt(abs(eigenvalues[0]))
                ver_rms = math.sqrt(abs(eigenvalues[1]))
                if ver_rms != 0:
                    elongation_num = math.sqrt(hor_rms/ver_rms)
                else:
                    elongation_num = 0
                # print("HERE is hor_rms in eign func: ", hor_rms)
                # print("HERE is ver_rms: ", ver_rms)
                # print("HERE is elong: ", elongation_num)
        
                return hor_rms, ver_rms, elongation_num  
    
    def is_line(self, hor_rms, elongation_num):
        """!@brief Determines if a cluster is line shaped based on horizontal RMS and elongation number.
            @return True if the cluster is line shaped, False otherwise."""
        if hor_rms > self.hor_rms_threshold or elongation_num > self.elongation_num_threshold:
            # print("HERE is hor_rms fFOR LINE ", hor_rms) 
            # print("HERE is hor_rms fFOR LINE ", elongation_num) 
            return True

        return False

    #---Fork 3 called by centriod_finder---
    def kd_tree_PCA(self, n_clusters):
        """!@brief Creates kd_trees starting at the centroids of the mid-zlice to link clusters across z-slices, then performs PCA.
            @detail To clarify, the kd_trees is made only from clustered points in all z-slices. This is done to increase resistance to noise for when PCA is applied to these point neightborhoods
            @see PCA_make_lines"""

        # print("HERE is txt dict: ", self.text_dict)
        self.kd_tree_PCA_done = False
        if len(self.clustered_cloud_list) > 0:
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())

            #Adding z values to the clustered cloud list, to make a 3D point cloud for KDTree and PCA, appending to all_pancakes list
            all_pancakes = []
            for i in range(self.pancake_stacks):
                # print("HERE is i: ", i)
                # print("HERE is clustered_cloud_list: ", self.clustered_cloud_list)
                curr_z = self.pancake_start + (self.pancake_gap * i)
                num_rows = np.shape(self.clustered_cloud_list[i])[0] 
                z_array = np.ones((num_rows,1)) * curr_z
                # print("z_array: ", z_array)
                # print("z_array shape: ", z_array.shape)
                all_pancakes.append(np.append(self.clustered_cloud_list[i],z_array,axis = 1))
                # TODO can optimize heavily if you just append xyz  of fullxyz[label boolean mask] instead of saving lists of x, y, then vstacking , do late r later

                # print("HERE is all_panacakes: ", all_pancakes)

            #V stacks all pancakes verically, to give KDtree all points from all pancakes
            self.all_vpancakes = np.vstack(all_pancakes)

            # print("HERE is all_vpancakes shape: ", self.all_vpancakes.shape)
            
            # print("HERE is self.mid_z_dict: ", self.mid_z_dict)

            if n_clusters > 0 and len(self.mid_z_dict):
                # print("INSIDE kd_tree_PCA, after checking n_clusters and mid_z_dict")
                for i in range(n_clusters): #TODO mayber change back to -1???
                    tree = KDTree(self.all_vpancakes, leaf_size =self.leaf_size)
                    clust_name = f"Cluster {i}"

                    # print("HERe is n_clusters: ", n_clusters)
                    # print("HERE is clust_name: ", clust_name)
                    # print("HERE is z_mid_dict: ", self.mid_z_dict)
                    centriod = np.array([self.mid_z_dict[clust_name]["xmean"], self.mid_z_dict[clust_name]["ymean"], self.mid_height]) #self.mid_height is a const in init

                    # print("HERE is CENTRIOD: ", centriod)
                    # print("HERE is vpanackes: ", all_vpancakes)
                    # print("HERE is vpanackes shape: ", all_vpancakes.shape)

                    # print("HERE is num_pts for clust looking at rn: ", self.mid_z_dict[clust_name]["num_pts"])
                    dist, ind = tree.query(centriod.reshape(1,-1), k = self.mid_z_dict[clust_name]["num_pts"]*self.pancake_stacks)

                    neighbors = self.all_vpancakes[ind[0]]

                    # print("HERE is neighbors: ", neighbors)
                    # print("HERE is neighbors shape: ", neighbors.shape)
                    # print("\n")
                    centered_neighbors = neighbors - centriod
                    pca = PCA(n_components=3)
                    fitted = pca.fit(centered_neighbors)

                    fitted_comps = np.abs(fitted.components_)
                    # print("Here is the PCA comps: ", fitted_comps)

                    all_z = fitted_comps[:, 2]
                    max_z_index = np.argmax(all_z)
                    z_eig = fitted_comps[max_z_index]

                    #Putting values in visualization dict
                    all_x = fitted_comps[:, 0]
                    max_x_index = np.argmax(all_x)
                    x_eig = fitted_comps[max_x_index]

                    all_y = fitted_comps[:, 1]
                    max_y_index = np.argmax(all_y)
                    y_eig = fitted_comps[max_y_index]

                    # print("HERE is clust_name in kd_tree_PCa: ", clust_name)
                    # print("INSIDE kd_tree_PCA")

                    self.text_dict[clust_name]["x_eig"] = x_eig
                    self.text_dict[clust_name]["x_index"] = max_x_index
                    self.text_dict[clust_name]["y_eig"] = y_eig
                    self.text_dict[clust_name]["y_index"] = max_y_index
                    self.text_dict[clust_name]["z_eig"] = z_eig
                    self.text_dict[clust_name]["z_index"] = max_z_index

                    

                    # print("HERE is max_z_index: ", max_z_index)
                    # print("HERE is z_eig: ", z_eig, "\n")

                    is_vertical_flag = self.is_vertical(z_eig)
                    if is_vertical_flag:
                        #Setting up list of centriods for persistence
                        if self.is_odom:
                            #Calculate values for hasbeenhere
                            drone_x = self.latest_pos.x
                            drone_y = self.latest_pos.y

                            xmean = self.mid_z_dict[clust_name]["xmean"]
                            ymean = self.mid_z_dict[clust_name]["ymean"]

                            x = xmean - drone_x
                            y = ymean - drone_y

                            angle = math.atan2(y,x)
                            
                            #TODO get angles from np.atan2(y,x) , make sure it works and gives you a 0 to 2pi  or -pi to pi once more
                            #TODO in hasbeen, make paramterization of lines and solves for the intersection, then you solve for how far it is from the two centroids (the parameter t)
                            #TODO now you do the conditional to make sure its not too damn far
                            #TODO also consider edge case with parallel stuff --> just dont consider it lol

                            centriod_xy = np.array([xmean, ymean, angle])
                            self.persistence_list.append(centriod_xy)


                        #TODO put centroids in a list, then make another func to be called in on_timer. This func is to make a numpy array from this list and vstack it. Use the vectorized np.unique on this, to get jus tthe uniq centriods. Then use this as 
                        #a guide for bool mask.sum() to count how many times it in there

                        #Or use np.unqie but with return_counts = True. use vectozied np.uniqe though

                    self.PCA_make_lines(z_eig, centriod, is_vertical_flag)

                self.kd_tree_PCA_done = True
            else:
                print("No clusters found in mid z-slice, or mid_z_dict is empty")
                self.kd_tree_PCA_done = False

    def PCA_make_lines(self, z_axis, centroid, is_vertical_flag):
        """!@brief Creates a of lines consistening of 20 points, with a length of 10. These lines represents the Z Principal Component passed in"""
        z_axis = abs(z_axis)
        z_basis = z_axis / np.linalg.norm(z_axis)

        #Creates a line of length 10, with 20 points. Adding a new axis makes it a col vector, to allow for broadcasting.
        length = 10
        line = np.linspace(0,length,20)[:,np.newaxis]

        #Performs broadcasting to the (20,1) and (3,) vectors, to allow for use of vectozied element-wise multiplication. Centroid it added so the line starts at the correct tree.
        curr_line = line * z_basis + centroid
        # print("HERE is curr_line: ", curr_line)

        #Whether the line passing verticality test, append it to the corresponding publishing list
        if is_vertical_flag: 
            self.pub_line_list.append(curr_line)
            # print("HERE is pub line list")
        else:
            self.pub_not_line_list.append(curr_line)

    def is_vertical(self, z_eig):
        """!@brief Determines if the angle of the Z Principal Component overcedes a certain threshold.
            @return A flag whether the angle threshold is passed or not."""
        dot = np.dot(z_eig, np.array([0,0,1]))
        z_eig_mag = np.linalg.norm(z_eig)
        angle = np.arccos(dot / z_eig_mag) * (180 / np.pi)
        # print("HERE is angle: ", angle)
        if angle < self.angle_threshold :  # Adjust the threshold as needed
            return True
        return False


#----------------------------------------------------------------Persistent Timer Thread(data dependant on centroid finder, indirectly on_timer)-----------------------------------------------------
    def persistence_timer(self, event):
        """!@brief This function is called every self.persistence_dur seconds. Calls persistence().
            @see persistence"""
        with self.lock:
            self.persistence()
            self.persistence_list.clear() #Clear the list, so new new persistence data is refreshed every self.persistence_dur sec

    def persistence(self):
        if len(self.persistence_list):
            #Vertically stack persistence_list, (N,2). Col's x, y centroids
            vpersist = np.vstack(self.persistence_list)
            # print("HERE vpersist shape: ", vpersist.shape)

            #Quantize the centroids, to allow for similarity checks later
            quantized = np.floor( vpersist / self.tol) * self.tol

            #Find the uniqe centroids that appear over self.persistence_dur. Also get the # times appears, for persistence checking. Done on quantized centroids to avoid high precision dec nums tricking persistence.
            _, inx, counts = np.unique(quantized, return_index = True, return_counts = True, axis = 0) #Choosing do regular np.unique since vpersist only (~100~,2)

            #Frequencies is of shape (N,3). Col's x, y, count of unique centroids
            frequencies = np.column_stack((vpersist[inx], counts))
            # print("freq: ", frequencies)

            #Finding the unique centroid with the highest count. #TODO May replace with highest possible count in self.persistence_dur.
            max_count = self.max_pers_counts #TODO fixing the coutn col cuz now the 4th one
            self.persistence_freq = max_count * self.freq_percent #This is our count threshold

            #Construting a boolean mask of counts that pass our count threshold. This is applied to frequencies to get the "persisted" centriods.
            persist_mask = frequencies[:,3] > self.persistence_freq
            persisted = frequencies[persist_mask]
            # print("persisted: ", persisted)

            #Quantizing persisted to allow for more silimarity checks for bookkeeping
            qpersisted = np.column_stack((np.floor(persisted[:,0:2] / self.tol) * self.tol, persisted[:,3])) #Adding the count col. back on after quantization

            print("HERE is qpersisted, fresh q: ", qpersisted)
            
            #Checking if our numpy array keeping track of trees been at (quantizied) is populated (done in dist_to_goal)
            # if self.qbeen.size > 0:
            print("HERE is self.qbeen: ", self.qbeen)
            #Checking is our quantized persisted centroids have alreadly been visited before. Quantized since we are doing similarity checks.
            in_mask = np.isin(qpersisted[:,0:2], self.qbeen).all(axis = 1) #.all(axis = 1) allows np.isin to look through rows #TODO using the false hits on notin_mask, add logic to if the counts better replace
            print("here is the fresh in_mask foor fresh scans", in_mask)
            # if self.qbeen.size > 0:
            #     #TODO Quantize?
            #     in_all_mask = np.isin(self.all_persisted_array, self.qbeen).all(axis = 1)
            # else:
            #     in_all_mask = np.ones_like(self.all_persisted_array, dtype=bool)

            #Ensuring that qpersisted, and persisted centriods are ones not visited before. Adding this as a 1/0 col at the end. (N,4). Col's x, y, angle, count, been.
            beencol = in_mask.T #or in_all_mask.T
            qpersisted = np.column_stack((qpersisted, beencol))
            persisted = np.column_stack((persisted, beencol))
    
            self.persist_bookkeeping(qpersisted, persisted)

            #TODO call hespanha's func, before seeing if been here, beacuse this func might add to qbeen
            self.hasbeenhere()

            # print("HERE is qpersisted: ", qpersisted)
            #Filtering self.all_persisted_array with centroids alreadly visited. This ensures a global list with only unvisited places is given to cost_map.
            #TODO get rid of b/c
            # if self.qbeen.size > 0:
            #     quantized_allp = np.floor(self.all_persisted_array / self.tol) * self.tol
            #     notin_mask = np.isin(quantized_allp, self.qbeen, invert = True).all(axis = 1)
            #     self.all_persisted_array = self.all_persisted_array[notin_mask]

            #     print("HERE is self.all_persisted_array after notin: ", self.all_persisted_array)

            self.scoring_func(self.all_persisted_array)


            #Publishing stuff:
            const_z_height = np.ones((self.all_persisted_array.shape[0], 1)) * 1.67
            self.pub_persisted_array = np.column_stack((self.all_persisted_array[:, 0:2], const_z_height))
            # print(self.pub_persisted_array)

    def hasbeenhere(self):
        if len(self.all_persisted_array):
            #Direcctions of the angles in polar
            dirs = np.column_stack((np.cos(self.all_persisted_array[:,2]), np.sin(self.all_persisted_array[:,2])))

            #chooses the upper triangle 1 diag above the main diag, to choose which where i and j are pairs we check against each other, wihtout repeating ourselves
            i, j = np.triu_indices(self.all_persisted_array.shape[0],k=1) #i and j are lists

            #Indexes self.all_persisted_array for the cenriod x,y, and only at the trianlge indices to creates unique pairs for all poss lines
            p1 = self.all_persisted_array[:,0:2][i]
            p2 = self.all_persisted_array[:,0:2][j]

            #Indexes the polar directions only at triangle indices
            d1 = dirs[i]
            d2 = dirs[j]

            #Construct the linear system to solve, finding the parameters t1 and t2 for all poss lines
            matrixA = np.stack((d1, -d2), axis = -1) #d1 and -d2 are placed as a tensor, shape is (i or j, 2, 2)
            b = p2 - p1

            #Finding the det, if it is 0 then there is no intersection (no sol), thus should not be included
            det = matrixA[:, 0, 0] * matrixA[:, 1, 1] - matrixA[:, 0, 1] * matrixA[:, 1, 0]

            #Create boolean mask for non par. lines
            nonpar = np.abs(det) > 1e-10


            #parrallel case, do min distance from d1 vector normal
            parallel = ~nonpar
            d1par = d1[parallel]
            p1par = p1[parallel]
            p2par = p2[parallel]
            n = np.column_stack((-d1par[:, 1], d1par[:, 0]))

            parallel_dist = np.abs(np.sum((p2par - p1par) * n, axis=1))

            par_mask = parallel_dist < 0.67 #TODO MAKE ME GLOBAL TUNABLE BUDDY 



            # print("HERE is nonpar boolean mask: ", nonpar)


            #nonparallel case, do solve for intersection legnth
            #Creating empty numpy area to hold the parameters of the lines we are going to solve for
            t = np.full((len(i),2), np.nan)

            #Masking t by nonpar to make right size. Then solving linear system to find parameters of the line (len of line for polar, r)
            t[nonpar] = np.linalg.solve(matrixA[nonpar], b[nonpar])

            # print("HERE is len of lines: ", t)

            #Creating another boolean mask for valid lines
            valid = (nonpar) & (t[:,0] >=-2) & (t[:,1] >= -2) & (t[:,0] <= self.linelen) & (t[:,1] <= self.linelen)

            print("HERE is p1[valid]: ", p1[valid])
            print("HERE is p2[valid]: ", p2[valid], "\n")

            self.make_lines()

            #TODO make an not is in mask for1st two col of self.allpersisited array and p1, this we then filter self.persisited array with yay
            # self.all_p1 = np.vstackp1[valid or par_mask]
            p1par_allmask = np.vstack((p1[valid], p1par[par_mask]))

            notin_mask = np.isin(self.all_persisted_array[:,0:2], p1par_allmask, invert = True).all(axis = 1)
            print("I AM HERE HAVING")
            self.all_persisted_array = self.all_persisted_array[notin_mask]


            #TODO MAYBE: right after, see if the intersected mask, the correlated one in p2, is intersecting any other points, as that will mean probably that centroid also same tree
            #TODO or just go in order and combine the labels that are of the same tree

    def persist_bookkeeping(self, qpersisted, persisted):
        """!@brief Checks incoming persisted are alrealdy in the bookkeeping numpy array. If not, they are added to"""
        #self.all_persisted_array is a global persisted numpy array. Reminder: (N,4). Col's x, y, angle, count, been.
        print("HERE is self.all_persisted_array BEFORE: ", self.all_persisted_array)

        if persisted.size != 0 :
            persisted = np.trunc(persisted * self.trunc_factor) / self.trunc_factor

        if self.all_persisted_array.shape == (0,):
            self.all_persisted_array = persisted
        else:
            #Checking if quantized persisted are alreadly in quantized self.all_persisted_array. Note: We use quantized since we are doing simliarity checks.
            notin_mask = np.isin(qpersisted[:,0:2], np.floor(self.all_persisted_array / self.tol) * self.tol, invert = True).all(axis = 1)
            if qpersisted[:,0:2][notin_mask].size != 0:
                print("HERE is qpersisted[:,0:2] notin: ", qpersisted[:,0:2][notin_mask])
                self.all_persisted_array = np.vstack((self.all_persisted_array, persisted[notin_mask])) #Adding on persisted not alreadly in 

        print("HERE is self.all_persisted_array AFTER: ", self.all_persisted_array)

    def scoring_func(self, persisted_array_all):
        """!@brief The next tree to visit is based on a linear combination of persistence and distance scores"""
        #TODO add score for zig zag waypts, and condiitonal to defult to zig zag path is nothing in self.all_persisted_array
        if persisted_array_all.size == 0:
            print("--------I AM NOT HAVING TRESS!--------")

        # qpersisted_array = np.floor(persisted_array / self.tol) * self.tol 

        not_been_mask = persisted_array_all[:,4] == 0
        print("not beeen here yet mask in the scoring: " , not_been_mask)
        persisted_array = persisted_array_all[not_been_mask]
        print("not beeen here persisted array fro scoring " , persisted_array) 

        persisted_counts = persisted_array[:, 3] #TODO changed to 4th col
        persisted_scores = (persisted_counts / (self.max_pers_counts)) * self.persisted_scores_weight #Persisted score is based on what the count is divided by the maximum count (see self.max_pers_counts)
        # print("HERE is count_scores: ", persisted_scores)

        if self.is_odom:
            drone_x = self.latest_pos.x
            drone_y = self.latest_pos.y

            x_dist = persisted_array[:, 0] - drone_x
            y_dist = persisted_array[:, 1] - drone_y

            xy_dist = np.column_stack((x_dist, y_dist))

            norms = np.linalg.norm(xy_dist, axis = 1)
            norms_score = (norms/np.max(norms)) * self.norms_scores_weight #Dist score is normalized to the max distance. Smaller dist score is betteer
            # print("HER is norm_score: ", norms_score)

            #Assesment, whichever linear combination is highest 
            assesment = persisted_scores - norms_score
            # print("HERE is asses: ", assesment)

            #Finding the index of the max_score, this will be the tree we go to
            max_index = np.argmax(assesment)
            # print("HERE is max_indx: ", max_index)
            print("HERE IS Where to go: ", persisted_array[max_index, 0:2])
            to_go = persisted_array[max_index, 0:2] #to_go is NOT quantized, since it is an actual place to go to.
            self.to_go(to_go)

            self.dist_to_goal(to_go) #TODO maybe put somehwere else where updated more than 3 secs? MAybe fine b/c assention

    def to_go(self, to_go):
        """!@brief Publishes a dot for the centriod to go to, as well as a position msg for SUPER
            @details Notice that to_go is not quantized. We want maximum precision to avoid collision."""
        #Creating message to publish
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        #Rviz Point
        point = PointStamped()
        point.header = header
        point.point.x = to_go[0]
        point.point.y = to_go[1]
        point.point.z = 2.67

        self.pub_to_go_pt.publish(point)

        #SUPER
        msg = PoseStamped() 
        msg.header = header

        # print(self.waypt_index)
        msg.pose.position.x = to_go[0]
        msg.pose.position.y = to_go[1]
        msg.pose.position.z = 0.25
        msg.pose.orientation.w = 1.0

        self.pub_super.publish(msg)

    def dist_to_goal(self, to_go):
        """!@brief Calculates the distance from the current odometry position to the place to go to.
            @details """
        #Quantizing the centroid to go to, to allow for putting this in self.qbeen
        qto_go = np.floor(to_go / self.tol) * self.tol

        drone_x = self.latest_pos.x
        drone_y = self.latest_pos.y

        dist_x = drone_x - to_go[0]
        dist_y = drone_y - to_go[1]

        squared_sum = pow(dist_x, 2) + pow(dist_y, 2)

        distance = math.sqrt(squared_sum)

        #If the to_go has been reached, then add qto_go 
        # print("D: ", distance)
        if distance < self.goal_tol:
            # print("waypt reached")
            #Flip the been col value to 1 for the centriod we went to
            #TODO where is togo in self.all_ersiste... quantize it first?
            qall_persisted_array = np.floor(self.all_persisted_array / self.tol) * self.tol

            row,cols = np.where(qall_persisted_array[:,0:2] == qto_go)
            self.all_persisted_array[row, 4] = 1.0

            if self.qbeen.size > 0:
                # print("I AM NOT HAVING: ", self.qbeen)
                self.qbeen = np.vstack((self.qbeen, qto_go))

            else:
                # print("I AM HAVING self.qbeen: ", self.qbeen)
                self.qbeen = qto_go





        #TODO pass in the persisted numpy array. Then save the normalized counts as persisted scores. Then compute the distance from where rn (from odom) to the centroid of mid z-slice.

    def make_lines(self):
        """!@brief Creates a of lines consistening of 20 points, with a length of 10. These lines represents the Z Principal Component passed in"""
        vec = np.column_stack((np.cos(self.all_persisted_array[:,2]), np.sin(self.all_persisted_array[:,2])))

        if vec.shape != (0,):
            length = self.linelen # i thiiink thats what ths is
            line = np.linspace(0,length,50)[:,np.newaxis]

            # const_z_height = np.ones((vec.shape[0], 1)) * 0.67
            # self.pub_persisted_array = np.column_stack((self.all_persisted_array[:, 0:2], const_z_height))

            for i in range(vec.shape[0]):
                self.hlines.append(line * vec[i] + self.all_persisted_array[i,0:2])


            

        






        #Performs broadcasting to the (20,1) and (3,) vectors, to allow for use of vectozied element-wise multiplication. Centroid it added so the line starts at the correct tree.
        # z = np.ones(vec.shape[0],1) * 0.67
        # withz = np.cloumn_stack((self.all_persisted_array[:,0:2],z))

        # print("HERE is shape sdfhslfgdhsg: ", self.all_persisted_array[:,0:2])
        # print("HERE is shape ofmulticpation: ",(vec @ stacked_line))

        # self.hlines = (vec @ stacked_line) + self.all_persisted_array[:,0:2]
        # print("HERE is hlines: ", self.hlines)
        # print("HERE is hlines shape: ", self.hlines.shape, "\n")

        #Whether the line passing verticality test, append it to the corresponding publishing list
        #TODO publish

    #TODO Filtering func. This will call PCA func, take the outputed numpy array (either reutnr of make global idk yet), and pass into PCA func. This willl then
    #take the poential trees PCA said aren't trees, and filter the cand_trees (or make a new list) accordingly

#---------------------------------------------------------------------------------------------Publishing Thread--------------------------------------------------------------------------------
    def publ(self):
        """!@brief"""
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        if self.all_vpancakes.any() != None:
            #TODO uncomment and fix the can't concatinate error
            # print("HERE IS publish_list: ", self.publish_list)
            # stacked_pub_list = np.vstack(self.publish_list)
            cluster_cloud = make_pointcloud2_xyz32(header, self.all_vpancakes)
            self.pub_cloud.publish(cluster_cloud)

        # print("Before if state here the pub linke ist: ", self.pub_line_list)
        if len(self.pub_line_list):
            # print("HERE is self.pub_line_list: ", self.pub_line_list)
            line_stacked = np.vstack(self.pub_line_list)
            line_cloud = make_pointcloud2_xyz32(header, line_stacked)
            self.pub_line.publish(line_cloud)

        if len(self.pub_not_line_list):
            not_line_stacked = np.vstack(self.pub_not_line_list)
            not_line_cloud = make_pointcloud2_xyz32(header, not_line_stacked)
            self.pub_not_line.publish(not_line_cloud)

        if self.pub_persisted_array.shape != (0,):
            persisted_dots = make_pointcloud2_xyz32(header, self.pub_persisted_array)
            self.pub_persisted.publish(persisted_dots)

        # if self.hlines != None:
        if len(self.hlines):
            # print("HERE is hlines: ", self.hlines)
            line_stacked = np.vstack(self.hlines)
            z_ones = np.ones((line_stacked.shape[0],1)) * 0.67
            line_stacked_with_z = np.column_stack((line_stacked,z_ones))
            hline_cloud = make_pointcloud2_xyz32(header, line_stacked_with_z)
            self.pub_hespline.publish(hline_cloud)

        #TODO check if the text dict is len, then publish
        #TODO uncomment for text debugging
        # print("HERE is if kd_tree_PCA_done: ", self.kd_tree_PCA_done)
        # if len(self.text_dict) > 0:

        #     marker_array = MarkerArray()

        #     for cluster in enumerate(self.text_dict):
        #         marker = Marker()
        #         marker.header = header
        #         marker.ns = "text_messages"
        #         marker.id = cluster[0]  # Unique ID per text string
        #         marker.type = Marker.TEXT_VIEW_FACING
        #         marker.action = Marker.ADD

        #         # print("HERE is text dict: ", self.text_dict, "\n")
        #         clus_num = cluster[1] #API have to do, 0 is index
        #         # Position of the text in 3D space
        #         marker.pose.position.x = self.text_dict[clus_num]["xmean"]
        #         marker.pose.position.y = self.text_dict[clus_num]["ymean"]
        #         marker.pose.position.z = 10
        #         marker.pose.orientation.w = 1.0
                
        #         # Text scale/size (Z controls height of capital letters)
        #         marker.scale.z = 0.15

        #         # Text color
        #         marker.color.r = 0
        #         marker.color.g = 0
        #         marker.color.b = 1.0
        #         marker.color.a = 1.0

        #         # print("INSIDE PUBLISH")
        #         # print(f"HERE is self.text_dict[{clus_num}]: ", self.text_dict[clus_num])
        #         # print("\n")

        #         hor_rms = round(self.text_dict[clus_num]["hor_rms"], 4)
        #         ver_rms = round(self.text_dict[clus_num]["ver_rms"], 4)
        #         elong = round(self.text_dict[clus_num]["elongation_num"], 4)

        #         x_eig = np.round(self.text_dict[clus_num]["x_eig"], decimals=4)
        #         x_index = self.text_dict[clus_num]["x_index"]
        #         y_eig = np.round(self.text_dict[clus_num]["y_eig"], decimals=4)
        #         y_index = round(self.text_dict[clus_num]["y_index"], 4)
        #         z_eig = self.text_dict[clus_num]["z_eig"]
        #         z_index = np.round(self.text_dict[clus_num]["z_index"], decimals=4)
                
        #         marker.text = f"hor_rms: {hor_rms}, ver_rms: {ver_rms}, elongation: {elong}, \nx_eig: {x_eig}, x_index: {x_index}, \ny_eig: {y_eig}, y_index: {y_index}, \nz_eig: {z_eig}, z_index: {z_index}"
        #         marker.lifetime = rospy.Duration(0.1)  # Refresh duration
                
        #         marker_array.markers.append(marker)

        #     self.pub_text.publish(marker_array)

        # if len(self.PCA_zaxis_list):
        #     marker = Marker()
        #     marker.header.frame_id = "world"
        #     marker.header.stamp = rospy.Time.now()
        #     marker.ns = "lines"
        #     marker.id = 0
        #     marker.type = Marker.LINE_LIST
        #     marker.action = Marker.ADD

        #     # Line width
        #     marker.scale.x = 0.05 

        #     # Color (Red, fully opaque)
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.color.a = 1.0

        #     point_list = []
        #     for z_axis in self.PCA_zaxis_list:
        #         x = z_axis[0]
        #         y = z_axis[1]
        #         z = z_axis[2]

        #         p = Point(x=x, y=y, z=z)

        #         point_list.append(p)

        #     marker.points = point_list

        #     self.pub_line.publish(marker)

    
        # if len(self.cand_trees):
        #     all_e_arrays = np.vstack(self.cand_trees)

        #     all_points = np.column_stack((all_e_arrays[:, 4:6], all_e_arrays[:, 3]))
        #     beacons = self.make_pointcloud2_xyz32(header, all_points)

        #     self.pub_beacon.publish(beacons)

    

def main():
    rospy.init_node("TreeFinder") #Make the node

    TreeFinder().run()

if __name__ =="__main__":
    main()