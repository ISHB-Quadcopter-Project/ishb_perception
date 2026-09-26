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

from boustrophedon import *

from geometry_msgs.msg import PoseStamped

import warnings
# Suppress the specific NumPy multidimensional indexing warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*non-tuple sequence.*")




#TODO We want to do two things A) find poss trees B) sep class at least, goes up to poss tree, and estimates radius, and reonsiders dist from tree for cam, and reconsiders tree placements
#b would happen before and while going towards tree, 
#When state 3 happens (assecnetion), that's when we know for certain the rad of the tree, and can accurately place waypt,next waypoint can b determined

class TreeFinder:
    """!@brief Detects candidate trees from an accumulated point cloud, dedups repeat trees, and sequences visiting.

        @details Still in development. The four horsemen pipelines of Treefinder.py:
        -# **Cloud/odom ingestion** (@ref cloud_cb, @ref odom_cb) — buffers the latest point cloud and
           odometry, slicing the cloud into horizontal 'pancake' z-bands (@ref cut_cloud).
        -# **on_timer** (10 Hz) — clusters each z-band (@ref centroid_finder -> @ref clustering -> eigen/@ref is_line),
           links clusters across bands with a KD-tree and classifies vertical, trunk-like ones via PCA
           (@ref kd_tree_PCA -> @ref PCA_make_lines / @ref is_vertical), then publishes debug clouds (@ref publ).
        -# **persistence_timer** (every `persistence_dur` sec) — keeps only detections that recur often enough
           to trust and merges them into a running candidate array (@ref persistence -> @ref persist_bookkeeping),
           collapses repeat sightings of the same physical tree from different vantage points (@ref hasbeenhere),
           and scores/selects the next target (@ref scoring_func).
        -# **goal_timer / doggy_timer** — track progress toward the current target at goal timer (@ref dist_to_goal) and
           republish the goal (@ref to_go) if the drone appears stuck at doggy_timer.

        
        @see on_timer @see persistence_timer @see goal_timer @see doggy_timer"""
    def __init__(self): 
        self.lock = threading.Lock()

        #------------Pub vars-------------
        # Topic super takes
        self.pub_super = rospy.Publisher("/super/goal", PoseStamped, queue_size = 10) 

        #Publish beacon/waypoint for found trees
        self.pub_beacon = rospy.Publisher("/Beacons", PointCloud2, queue_size = 10)

        self.pub_cloud = rospy.Publisher("/Z_CLUSTERS", PointCloud2, queue_size = 10)

        self.pub_line = rospy.Publisher("LINES", PointCloud2, queue_size = 10)

        self.pub_hespline = rospy.Publisher("HESP_LINES", PointCloud2, queue_size = 10)

        self.pub_not_line = rospy.Publisher("NOT_LINES", PointCloud2, queue_size = 10)

        self.pub_persisted = rospy.Publisher("PERSISTED", PointCloud2, queue_size = 10)

        self.pub_intersect = rospy.Publisher("INTERSECT", PointCloud2, queue_size = 10)

        self.pub_to_go_pt = rospy.Publisher("TOGO", PointStamped, queue_size = 10)

        self.pub_text = rospy.Publisher('/TEXT', MarkerArray, queue_size=10)

        self.zztop = rospy.Publisher('ZZTOP', Marker, queue_size=10)



        """@par Tunable parameters
                Set in `__init__`; most flagged `#XXX tuning!!!` there.
        
                - __Clustering__
                  - `eps` (ROS param `~dbscan_eps`, default 0.25) — DBSCAN neighborhood radius.
                  - `min_samples` (`~dbscan_min_samples`, default 5) — min points to form a cluster.
                  - `hor_rms_threshold` (0.2) / `elongation_num_threshold` (1) — @ref is_line thresholds separating
                    line-shaped clusters (branches/ground) from trunk candidates.
                  - `angle_threshold` (25 deg) — @ref is_vertical cutoff for classifying a cluster's PCA axis as a
                    vertical trunk.
                  - `leaf_size` (80) — KDTree leaf size used in @ref kd_tree_PCA.
                - __Pancake z-slicing__ (see @ref cut_cloud)
                  - `pancake_stacks` (7) — number of horizontal z-bands scanned per cycle.
                  - `pancake_start` (5*0.067 m) — height of the lowest band.
                  - `pancake_gap` (1*0.067 m) — vertical spacing between bands.
                  - `pancake_thickness` (3*0.067 m) — thickness of each band.
                - __Persistence/dedup__
                  - `tol` (0.08 m) — quantization grid used for similarity checks in @ref persistence,
                    @ref persist_bookkeeping, @ref hasbeenhere, and `self.qbeen`.
                  - `freq_percent` (0.5) together with `persistence_dur` (3 s) — fraction of the maximum possible hit
                    count (`self.max_pers_counts`) a centroid must reach within this window to be trusted as a real
                    tree (see @ref persistence).
                  - `goal_tol` (1 m) — distance within which a target counts as reached (@ref dist_to_goal).
                  - `linelen` (2 m) / `backlen` (0.5 m) — sightline length bounds shared by the trunk-line
                    visualizations (@ref PCA_make_lines, @ref make_lines) and the intersection test in @ref hasbeenhere.
                - __Scoring__ (@ref scoring_func)
                  - `tree_const` (1.5), `persisted_scores_weight` (1), `norms_scores_weight` (4),
                    `per_waypt_weight` (0.02), `norm_waypt_weight` (2) — relative weights combining a candidate's
                    persistence count against its distance from the drone when picking the next target.
                - __Timer periods__
                  - `on_timer_dur` (0.1 s, ~10 Hz), `persistence_dur` (3 s), `goal_timer_dur` (0.25 s),
                    `doggy_timer_dur` (4 s) — periods of the four `rospy.Timer` callbacks.
                - __Debug plotting__
                  - `debug_plot` (ROS param `~debug_plot`, default False) — enables @ref _save_cluster_plot.
                  - `plot_period` (`~plot_period`, default 2.0 s) and `plot_dir` (`~plot_dir`, default
                    `~/ishb_ws/debug_plots`) — throttle and output directory for those plots.
        """
        #------------Pancake PARAMS------------- #XXX tuning!!!
        self.pancake_stacks = 7
        self.pancake_start = round(5 * 0.067, 5)  #5 #TODO make some sorta global arg from config?, or maybe a func that reads odom and updates it
        self.pancake_gap = round(1* 0.067, 5) #TODO This has to be small for new alg
        self.pancake_thickness = round(3 * 0.067, 5)
        self.mid_height = round(self.pancake_stacks/2) *self.pancake_gap + self.pancake_start #Not including in calcualtion b/c so small

        self.publish_list = deque(maxlen =self.pancake_stacks*1)

        #------------Sub vars-------------
        self.sub = rospy.Subscriber("/Cum_Cloud", PointCloud2, self.cloud_cb, queue_size = 10)
        self.latest_cloud = None
        self.processed_cloud_list = deque(maxlen = self.pancake_stacks*2)

        #Odom var to hold the x,y,z odom data
        self.latest_pos = None

        self.sub = rospy.Subscriber("/Odometry", Odometry, self.odom_cb, queue_size = 10)
        
        #Flag to see if there is available odom data to check dist_to_goal
        self.is_odom = False



        #------------PCA kd tree vars-------------
        self.hor_rms_threshold = 0.2 #Measure of how spread horizontally #XXX tuning!!!
        self.elongation_num_threshold = 1 #ranges 0-1, higher is more "circular" #XXX tuning!!!

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
        self.last_plot = None


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



        #------------Persistence PARAMS/vars-------------
        self.persistence_list = []
        self.tol = 0.08
        self.goal_tol = 1 #XXX tuning!!! 

        self.freq_percent = 0.375 #XXX tuning!!!

        self.persistence_dur = 3 #XXX tuning!!!
        self.on_timer_dur = 0.1
        self.goal_timer_dur = 0.25 #XXX tuning!!!
        self.doggy_timer_dur = 4 #XXX tuning!!!

        self.tree_const = 1.5 #XXX tuning!!!
        self.persisted_scores_weight = 1 #XXX tuning!!!
        self.norms_scores_weight = 4 #XXX tuning!!!

        self.per_waypt_weight = 0.02 #XXX tuning!!!
        self.norm_waypt_weight = 2 #XXX tuning!!!

        # self.keep_circle = 0.5 #XXX tuning!!!

        self.linelen = 2 #XXX tuning!!!
        self.backlen = 0.5 #XXX tuning!!!
        self.parallel_dist_threshold = 1 #XXX tuning!!!

        self.max_pers_counts = self.persistence_dur / self.on_timer_dur #Maximum possible counts is the duration of persistence, divided by how often you add to the persistence_list
        self.pub_persisted_array = np.zeros(0)

        self.all_persisted_array, row_ys = build_persisted_array()
        self.all_persisted_array[:,0]  = self.all_persisted_array[:,0] - 25

        self.waypoint_index = 0

        self.num_waypts = self.all_persisted_array.shape[0]

        self.cur_to_go = np.zeros(0)
        
        self.qbeen = np.zeros(0)

        self.hlines = []

        self.max_index = 0

        self.trunc_deci = 5
        self.trunc_factor = 10 ** 5

        self.all_p1 = np.zeros(0)

        self.intersect_pub_array = np.zeros(0)

        self.odom_list = []

        self.trees = np.zeros(0)

        self.curwaypt = np.zeros(0)

        self.assesment = np.zeros(0)

        self.glo_per_score = np.zeros(0)

        self.glo_norm_score = np.zeros(0)

        self.scoring_flag = True



        #------------Timer vars-------------
        rospy.Timer(rospy.Duration(self.on_timer_dur), self.on_timer)  # 10 Hz

        rospy.Timer(rospy.Duration(self.persistence_dur), self.persistence_timer)  # 1 Hz

        rospy.Timer(rospy.Duration(self.goal_timer_dur), self.goal_timer)

        rospy.Timer(rospy.Duration(self.doggy_timer_dur), self.doggy_timer)
        


    def run(self):
        """!@brief Blocks the main thread from ending
            @details All actual work happens in the subscriber callbacks and rospy.Timer threads registered in __init__; this just keeps the node alive."""
        rospy.spin()

    
#---------------------------------------------------------------------------------------Subscriber Thread-----------------------------------------------------------------------------------------
    def odom_cb(self, msg):
        """!@brief Callback function for the /Odometry topic
            @details Updates the latest odometry position and sets the is_odom flag to True. Odom data is stored in a list for the odom_watchdog to check if the drone is moving. It's also used in building all_persisted_array to find the bearing angle
            @note self.lock is used to ensure other areas of code using odom data don't get partial data, as this callback is in a separate thread
            @param msg The Odometry message received from the /Odometry topic"""
        
        with self.lock:
            self.latest_pos = msg.pose.pose.position
            self.odom_list.append(self.latest_pos) #Add odom data to list for odom_watchdog
            self.is_odom = True # latest_pos odom should be set by now

    def cloud_cb(self, msg):
        """!@brief Callback function for the /Cum_Cloud topic.
            @details Adds clouds with specified z-ranges using cut_cloud to a list for centroid_finder, using cloud_to_xyz
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
            @details Uses boolean mask for specified z-range, to "cut" the cloud. A bit-packed key is created for the x,y coordinates, to find make finding unique x,y coordinates faster. This part is similar to Accumulator.Accumulator.down_cloud. self.pancake_thickness is the thickness of the pancake, and is global not passed as a parameter
            @param uncut_cloud The numpy array point cloud to cut
            @param z_mid The middle height of the z-range to cut
            @return A numpy array of shape (N, 2) containing the x,y coordinates of the points in the specified z-range
            @see cloud_cb"""

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
        self.publish_list.append(cut_cloud[first])


        #Return cut_cloud with indices that only include uniqe x,y's
        return cut_cloud[first,0:2] 


#--------------------------------------------------------------------------on_timer Thread (that sub and pub both depend on)-------------------------------------------------------------------------------
    def on_timer(self,event):
        """!@brief Timer callback to call centroid_finder on each z-slices. Calls publ too.
            @details This will pass in a bool flag to centroid_finder dictating whether it is the mid z-slice. If kd_tree_PCA is done, based on a flag, then relevant list and dicts for this class's operations are cleared to ensure data is refreshed.
            @param event An object of TimerEvent, automatically created every time rospy.Timer fires.
            @see centroid_finder @see publ"""

        i = 1

        self.mid_count = 0 #reset the mid_count for kd_tree_PCA
        with self.lock:
            if len(self.processed_cloud_list):
                half = self.pancake_stacks / 2
                mid_num = math.floor(half + 0.5)
                for pancake_num in range(self.pancake_stacks):
                    is_mid = False

                    if i == mid_num: #Checking if at the mid z-slice
                        is_mid = True

                    self.centroid_finder(pancake_num, is_mid)

                    i += 1

                    
                        

                if self.kd_tree_PCA_done == True:
                    self.publ()
                    self.pub_line_list.clear()
                    self.pub_not_line_list.clear()
                    self.text_dict.clear()
                    self.mid_z_dict.clear() #Clear dict for mid slice, before poulate again with new clustering
                    self.clustered_cloud_list.clear()

    def centroid_finder(self, which, is_mid):
        """!@brief Calls clustering on a specified z-slice. If it is the middle z-slice and not line shaped, does a num_pts hard cap filter, and saves RMS, elongation, centroid, amount of points for kd_tree_PCA and also publishing text.
            @see is_line @see clustering @see kd_tree_PCA @see on_timer
            @param which An integer value representing which pancake is to be clustered. 
            @todo Line 355, 1000 is used as a crazy number, and maybe will need to be changed when real lidar comes. return is also artifact of e_array stuff"""

        self.kd_tree_PCA_done = False
        if len(self.processed_cloud_list) > which:
            processed_cloud = self.processed_cloud_list[which]
            if len(processed_cloud):
                labels, n_clusters = self.clustering(processed_cloud, which)

                if n_clusters > 0:
                    self.clustered_cloud_list.append(self.xy[labels != -1])
                    for clustnum in range(n_clusters):
                        curr_clust = self.xy[labels == clustnum]

                        #A 0 or 1 pt cluster has no shape to measure, and would otherwise
                        #register as a tree candidate with rms 0. Not a real cluster, skip.
                        if curr_clust.shape[0] < 2:
                            continue

                        xmean = np.mean(curr_clust[:,0])
                        ymean = np.mean(curr_clust[:,1])

                        #Getting relevant cluster info from eigen func
                        hor_rms, ver_rms, elongation_num = self.eigen(curr_clust, xmean, ymean)

                        clust_name = f"Cluster {self.mid_count}"

                        #Populate mid_z_dict if at mid z-sclie
                        if is_mid and not self.is_line(hor_rms, elongation_num):
                            num_pts = curr_clust.shape[0]

                            if num_pts < 1000: #Checking if the cluster is absurd

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

                                #Populating alr instantiated numpy array in mem. This array holds cluster info for all clusters in a z "pancake" slice
                                # e_array[clustnum] = [hor_rms, ver_rms, elongation_num, which,xmean,ymean]

                    if which == self.pancake_stacks - 1: 
                        self.kd_tree_PCA(self.mid_count)
                return None

    #---Fork 1 called by centriod_finder---
    def clustering(self, points, which): 
        """!@brief Clusters a point cloud using DBSCAN and returns the labels and number of clusters.
            @details calls _save_cluster_plot for debugging purposes(if the boolean in debug_plot is true).
            @return The labels and number of clusters
            @see centroid_finder @see _save_cluster_plot
            @todo May want to remove the rospy.loginfo eventually, once finished IRL testing and tuning"""
        
        
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
            now = rospy.Time.now()
            self._save_cluster_plot(self.xy, labels, n_clusters, now ,which)                

        return labels, n_clusters

    def _save_cluster_plot(self, xy, labels, n_clusters, stamp ,which):
        """!@brief Saves a plot of the clustered point cloud for debugging purposes.
            @param xy [in] The (N,2) array of x,y points that were clustered
            @param labels [in] DBSCAN cluster label per point in xy (-1 marks noise)
            @param n_clusters [in] Number of non-noise clusters found
            @param stamp [in] ROS time used to timestamp the output filename
            @param which [in] Index of the pancake z-slice being plotted, used to compute its height for the plot title/filename
            @note Only called when self.debug_plot is enabled; writes a PNG to self.plot_dir
            @todo so very spammy holy cow
            @see clustering"""
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
            #np.cov ignores rowvar=False on a single-row array, collapsing to a 0-d
            #scalar that np.linalg.eig rejects. Need >= 2 points for a covariance anyway.
            if curr_clust.shape[0] >= 2:

                normalized = curr_clust - np.array([xmean,ymean]) #To do it at origin

                cov_matrix = np.cov(normalized, rowvar = False)

                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

                hor_rms = math.sqrt(abs(eigenvalues[0]))
                ver_rms = math.sqrt(abs(eigenvalues[1]))
                if ver_rms != 0:
                    elongation_num = math.sqrt(hor_rms/ver_rms)
                else:
                    elongation_num = 0

                return hor_rms, ver_rms, elongation_num

            #Degenerate cluster (0 or 1 pts): no spread to measure
            return 0.0, 0.0, 0.0

    def is_line(self, hor_rms, elongation_num):
        """!@brief Determines if a cluster is line shaped based on horizontal RMS and elongation knobs.
            @param hor_rms [in] Horizontal RMS spread of the cluster from eigen()
            @param elongation_num [in] Elongation ratio of the cluster from eigen(), higher means less circular
            @return True if the cluster is line shaped (e.g. a branch/ground feature rather than a tree trunk), False otherwise.
            @see centroid_finder"""
        if hor_rms > self.hor_rms_threshold or elongation_num > self.elongation_num_threshold: 
            return True

        return False

    #---Fork 3 called by centriod_finder---
    def kd_tree_PCA(self, n_clusters):
        """!@brief Creates kd_trees starting at the centroids of the mid-zlice to link clusters across z-slices, then performs PCA.
            @detail To clarify, the kd_trees is made only from clustered points in all z-slices. This is done to increase resistance to noise for when PCA is applied to these point neightborhoods. Flag used to avoid errors while publishing prints to rviz
            @param n_clusters [in] Number of clusters found in the mid z-slice (self.mid_z_dict), each looked up by name and expanded into a local 3D neighborhood for PCA
            @return None. Side effects: populates self.text_dict per cluster with its principal-axis (eigenvector) info, appends candidate centroids to self.persistence_list when the cluster reads as vertical, and sets self.kd_tree_PCA_done to signal on_timer/publ that fresh results are ready
            @ todo Can optimize heavily(if tree arrays get super big) if you just append xyz  of fullxyz[label boolean mask] instead of saving lists of x, y, then appending, and then appending , do later later
            @see PCA_make_lines @see is_vertical @see centroid_finder"""

        # print("HERE is txt dict: ", self.text_dict)
        self.kd_tree_PCA_done = False
        if len(self.clustered_cloud_list) > 0:
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())

            #Adding z values to the clustered cloud list, to make a 3D point cloud for KDTree and PCA, appending to all_pancakes list
            all_pancakes = []
            for i in range(self.pancake_stacks):
                curr_z = self.pancake_start + (self.pancake_gap * i)
                num_rows = np.shape(self.clustered_cloud_list[i])[0] 
                z_array = np.ones((num_rows,1)) * curr_z
                all_pancakes.append(np.append(self.clustered_cloud_list[i],z_array,axis = 1))
                # TODO can optimize heavily(if tree arrays get super big) if you just append xyz  of fullxyz[label boolean mask] instead of saving lists of x, y, then appending, and then appending , do later later

            #V stacks all pancakes verically, to give KDtree all points from all pancakes
            self.all_vpancakes = np.vstack(all_pancakes)
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

                    self.text_dict[clust_name]["x_eig"] = x_eig
                    self.text_dict[clust_name]["x_index"] = max_x_index
                    self.text_dict[clust_name]["y_eig"] = y_eig
                    self.text_dict[clust_name]["y_index"] = max_y_index
                    self.text_dict[clust_name]["z_eig"] = z_eig
                    self.text_dict[clust_name]["z_index"] = max_z_index

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

                            centriod_xy = np.array([xmean, ymean, angle])
                            self.persistence_list.append(centriod_xy)

                    self.PCA_make_lines(z_eig, centriod, is_vertical_flag)

                self.kd_tree_PCA_done = True
            else:
                print("No clusters found in mid z-slice, or mid_z_dict is empty")
                self.kd_tree_PCA_done = False

    def PCA_make_lines(self, z_axis, centroid, is_vertical_flag):
        """!@brief Creates a of lines consistening of 20 points, with a length of 10. These lines represents the Z Principal Component passed for a certain tree candidate. This is mostly for printing/debugging in rviz
            @param z_axis [in] The cluster's Z principal-component eigenvector (from kd_tree_PCA), used as the line's direction
            @param centroid [in] The 3D point (x, y, mid_height) the line is anchored at
            @param is_vertical_flag [in] Whether is_vertical() classified this axis as vertical enough to be a tree trunk candidate; routes the line into pub_line_list (candidate) vs pub_not_line_list (rejected), for rviz visualization
            @see kd_tree_PCA"""
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
            @param z_eig [in] The Z principal-component eigenvector of a cluster's neighborhood, from kd_tree_PCA
            @return True if the angle between z_eig and the world Z axis is under self.angle_threshold degrees (i.e. the cluster looks like a vertical trunk), False otherwise.
            @see kd_tree_PCA"""
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
        """!@brief Aggregates the vertical-cluster centroids seen since the last persistence_dur window into stable, deduplicated tree/waypoint candidates and triggers scoring.
            @details Quantizes centroids to self.tol to absorb detection jitter, keeps only ones seen frequently enough (self.freq_percent of self.max_pers_counts) to be trusted as real, merges them into the running self.all_persisted_array via persist_bookkeeping, resolves duplicate detections of the same physical tree via hasbeenhere, then re-scores candidates via scoring_func.
            @see persist_bookkeeping @see hasbeenhere @see scoring_func @see persistence_timer"""
        if len(self.persistence_list):
            #Vertically stack persistence_list, (N,2). Col's x, y centroids
            # print("HERE is persistence list: ", self.persistence_list)

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
            persistence_freq = max_count * self.freq_percent #This is our count threshold

            #Construting a boolean mask of counts that pass our count threshold. This is applied to frequencies to get the "persisted" centriods.
            persist_mask = frequencies[:,3] > persistence_freq
            persisted = frequencies[persist_mask]
            # print("persisted: ", persisted)

            #Quantizing persisted to allow for more silimarity checks for bookkeeping
            qpersisted = np.column_stack((np.floor(persisted[:,0:2] / self.tol) * self.tol, persisted[:,3])) #Adding the count col. back on after quantization

            #Checking is our quantized persisted centroids have alreadly been visited before. Quantized since we are doing similarity checks.
            in_mask = np.isin(qpersisted[:,0:2], self.qbeen).all(axis = 1) #.all(axis = 1) allows np.isin to look through rows #TODO using the false hits on notin_mask, add logic to if the counts better replace

            #Ensuring that qpersisted, and persisted centriods are ones not visited before. Adding this as a 1/0 col at the end. (N,4). Col's x, y, angle, count, been.
            beencol = in_mask.T #or in_all_mask.T
            qpersisted = np.column_stack((qpersisted, beencol))
            persisted = np.column_stack((persisted, beencol))
            #book keeping to dedup, has been here to continue dedup, then score, no dups in scoring, when tuned right
            self.persist_bookkeeping(qpersisted, persisted)

            self.hasbeenhere()

            self.scoring_func(self.all_persisted_array)


            #Publishing stuff:
            const_z_height = np.ones((self.all_persisted_array.shape[0], 1)) * 1.67
            self.pub_persisted_array = np.column_stack((self.all_persisted_array[:, 0:2], const_z_height))

            # print(self.pub_persisted_array)
    
    def hasbeenhere(self):
        """!@brief Detects when two persisted tree centroids are really the same physical tree observed from different vantage points, and merges their visited ('been') flags.
            @details Each persisted tree row carries the bearing angle from the drone to the tree at detection time; treating that bearing as a sightline through the centroid, every pair of tree rows is tested for whether their sightlines intersect near both points (within [-self.backlen, self.linelen] of each line's parameter) or, if the sightlines are parallel, whether they pass within a small perpendicular distance of each other. Either condition is taken as evidence the two rows are the same tree seen from two positions, so their 'been' column is OR'd together in self.all_persisted_array. Also publishes debug geometry (perp/intersection point clouds) for rviz.
            @note Only runs once more persisted trees exist than fixed waypoints (self.num_waypts), since the first self.num_waypts rows of self.all_persisted_array are boustrophedon waypoints, not trees.
            @see persistence @see make_lines"""
        if len(self.all_persisted_array) > self.num_waypts:
            #Direcctions of the angles in polar
            waypts_mask = self.all_persisted_array[:,3] < 0
            not_waypts = ~waypts_mask
            trees = self.all_persisted_array[not_waypts]
            dirs = np.column_stack((np.cos(trees[:,2]), np.sin(trees[:,2])))

            #chooses the upper triangle 1 diag above the main diag, to choose which where i and j are pairs we check against each other, wihtout repeating ourselves
            i, j = np.triu_indices(trees.shape[0],k=1) #i and j are lists

            #Indexes trees for the cenriod x,y, and only at the trianlge indices to creates unique pairs for all poss lines
            p1 = trees[:,0:2][i]
            p2 = trees[:,0:2][j]

            #Indexes the polar directions only at triangle indices
            d1 = dirs[i]
            d2 = dirs[j]

            #Construct the linear system to solve, finding the parameters t1 and t2 for all poss lines
            matrixA = np.stack((d1, -d2), axis = -1) #d1 and -d2 are placed as a tensor, shape is (i or j, 2, 2)
            b = p2 - p1

            #Finding the det, if it is 0 then there is no intersection (no sol), thus should not be included
            det = matrixA[:, 0, 0] * matrixA[:, 1, 1] - matrixA[:, 0, 1] * matrixA[:, 1, 0]

            #Create boolean mask for non par. lines
            nonpar = np.abs(det) > 0.009


            #parrallel case, do min distance from d1 vector normal
            parallel = ~nonpar
            # print("parallel mask: ", parallel)
            d1par = d1[parallel]
            p1par = p1[parallel]
            p2par = p2[parallel]
            n = np.column_stack((-d1par[:, 1], d1par[:, 0]))
            parallel_dist = np.abs(np.sum((p2par - p1par) * n, axis=1))


            par_mask = parallel_dist < self.parallel_dist_threshold


            #nonparallel case, do solve for intersection legnth
            #Creating empty numpy area to hold the parameters of the lines we are going to solve for
            t = np.full((len(i),2), np.nan)

            #Masking t by nonpar to make right size. Then solving linear system to find parameters of the line (len of line for polar, r)
            t[nonpar] = np.linalg.solve(matrixA[nonpar], b[nonpar])

            #Creating another boolean mask for valid lines
            #Only compare rows solved above (nonpar); the rest are still NaN and would
            #trigger spurious "invalid value" warnings on comparison
            valid = np.zeros(len(i), dtype=bool)
            tnp = t[nonpar]
            valid[nonpar] = (tnp[:,0] >= -self.backlen) & (tnp[:,1] >= -self.backlen) & (tnp[:,0] <= self.linelen) & (tnp[:,1] <= self.linelen)

            self.make_lines()

            # self.all_p1 = np.vstackp1[valid or par_mask]
            #long list of all paired points that are deemed the same

            #For publishing if they intersect, pink dots
            p1par_all = np.vstack((p1[valid], p1par[par_mask]))
            p2par_all = np.vstack((p2[valid], p2par[par_mask]))

            # print("HERE is p1[valid]: ", p1[valid])
            # print("HERE is p2[valid]: ", p2[valid], "\n")
            # print("HERE is p1par_all: ", p1par_all)
            # print("HERE is p2par_all: ", p2par_all, "\n")

            intersect_pub_array = (p1par_all + p2par_all) / 2
            const_z_height = np.ones((intersect_pub_array.shape[0], 1)) * 2.67
            if self.intersect_pub_array.shape != (0,):
                self.intersect_pub_array = np.vstack((self.intersect_pub_array,np.column_stack((intersect_pub_array , const_z_height))))
            else:
                self.intersect_pub_array = np.column_stack((intersect_pub_array , const_z_height))


            
            # print("HERE is self.all_persisted_array BEFORE: ", self.all_persisted_array)

            # in_mask1 = np.isin(self.all_persisted_array[:,0:2], p1par_all).all(axis = 1)
            # in_mask2 = np.isin(self.all_persisted_array[:,0:2], p2par_all).all(axis = 1)
            # print("HERE Is the shape of inmask1,size shold b same as apa: ", in_mask1.shape)
            # print("HERE Is the shape of inmask2: ", in_mask2.shape, "\n")

            #TODO for loop time
            ind_list1 = []
            ind_list2 = []

            if p1par_all.size != 0 and p2par_all.size != 0:
                for x in range(p1par_all.shape[0]):
                    # if np.any(self.all_persisted_array == p1par_all[x]) and np.any(self.all_persisted_array == p2par_all[x]):
                    # print("HEREEE is p1par[x]: ", p1par_all[x])
                    # print("HEREEE is p2par[x]: ", p2par_all[x])

                    index1 = np.where(np.all(self.all_persisted_array[:,0:2] == p1par_all[x], axis=1))[0][0]
                    index2 = np.where(np.all(self.all_persisted_array[:,0:2] == p2par_all[x], axis=1))[0][0]
                    if index1 > 26 and index2 > 26:
                        ind_list1.append(index1)
                        ind_list2.append(index2)

            # print("HERE is in_list2: ", ind_list2)
            # print("size of allpersistedarray[indlist2] out o the loop", self.all_persisted_array[ind_list2, :])

            if self.all_persisted_array[ind_list1].size != 0 or self.all_persisted_array[ind_list2].size != 0:
                # print("HERE Is the shape of p1: ", self.all_persisted_array[ind_list1][:,4].shape)
                # print("HERE Is the shape of p2: ", self.all_persisted_array[ind_list2][:,4].shape, "\n")

                # print("HERE is p1 col: ", self.all_persisted_array[ind_list1][:,4])
                # print("HERE is p2 col: ", self.all_persisted_array[ind_list2][:,4])
                new_bool_col = np.logical_or(self.all_persisted_array[ind_list1][:,4], self.all_persisted_array[ind_list2][:,4])
                # print("HERE is new bool col: ", new_bool_col)

                self.all_persisted_array[ind_list1, 4] = new_bool_col.astype(np.float32)
                self.all_persisted_array[ind_list2, 4] = new_bool_col.astype(np.float32)

            # print("HERE is self.all_persisted_array AFTER: ", self.all_persisted_array, "\n")


            #TODO MAYBE: right after, see if the intersected mask, the correlated one in p2, is intersecting any other points, as that will mean probably that centroid also same tree
            #TODO or just go in order and combine the labels that are of the same tree

    def persist_bookkeeping(self, qpersisted, persisted):
        """!@brief Checks incoming persisted are alrealdy in the bookkeeping numpy array. If not, they are added to
            @param qpersisted [in] Quantized [x, y, count, been] rows (self.tol grid) for the trees that passed the frequency filter this window, used only for similarity comparisons against the quantized self.all_persisted_array
            @param persisted [in] The same rows as qpersisted but at full precision (truncated to self.trunc_deci decimals), appended to self.all_persisted_array when not already present
            @note Skips rows already known (matched by quantized x,y against non-waypoint rows of self.all_persisted_array) so the same physical tree isn't added twice.
            @see persistence"""
        #self.all_persisted_array is a global persisted numpy array. Reminder: (N,4). Col's x, y, angle, count, been.
        # print("HERE is self.all_persisted_array BEFORE: ", self.all_persisted_array)

        if persisted.size != 0 :
            persisted = np.trunc(persisted * self.trunc_factor) / self.trunc_factor

        #TODO 2) check if exact same size starrted with
        if self.all_persisted_array.shape == (self.num_waypts,):
            self.all_persisted_array = np.vstack((self.all_persisted_array, persisted))
        else:
            #Only bookkeeping on trees, not waypts
            not_waypts = self.all_persisted_array[:,2] > 0

            #Checking if quantized persisted are alreadly in quantized self.all_persisted_array. Note: We use quantized since we are doing simliarity checks.
            notin_mask = np.isin(qpersisted[:,0:2], np.floor(self.all_persisted_array[not_waypts] / self.tol) * self.tol, invert = True).all(axis = 1)
            if qpersisted[:,0:2][notin_mask].size != 0:
                # print("HERE is qpersisted[:,0:2] notin: ", qpersisted[:,0:2][notin_mask])
                #TODO Already-known centroids are never updated after their first match here - we just
                #keep whatever x,y,count was recorded the first time and drop every later re-detection
                #of the same tree. Need a way to fold in new detections instead (e.g. running average
                #of x,y, or keep the highest-count/most-confident observation) rather than trusting
                #only the first persisted hit.
                self.all_persisted_array = np.vstack((self.all_persisted_array, persisted[notin_mask])) #Adding on persisted not alreadly in

        # print("HERE is self.all_persisted_array AFTER: ", self.all_persisted_array)
        print("HERE is apa waypts: ", self.all_persisted_array[0:self.num_waypts,:])

    def scoring_func(self, persisted_array_all):
        """!@brief The next tree to visit is based on a linear combination of persistence and distance scores
            @param persisted_array_all [in] The full [x, y, angle, count, been] bookkeeping array (self.all_persisted_array), covering both boustrophedon waypoints (angle < 0) and detected trees
            @details Trees not yet visited are scored by persistence count (self.persisted_scores_weight) minus normalized distance from the drone (self.norms_scores_weight); the next unvisited waypoint gets its own fixed persistence/distance weighting so it can still win if no tree scores highly. The highest-scoring candidate is written to self.cur_to_go and self.scoring_flag is cleared so scoring is skipped until dist_to_goal() reports that goal reached.
            @note Only runs while self.scoring_flag is True and requires self.is_odom to compute distances; silently no-ops (aside from prints) otherwise.
            @bug self.scoring_flag exists specifically to gate this: without it, scoring_func would re-run and recompute self.cur_to_go on every persistence_timer tick before dist_to_goal() ever marks the current target as reached ('been' = 1), causing the drone to skip targets mid-approach and re-visit ones it already scored past.
            @see persistence @see dist_to_goal"""
        #TODO 3) make boool mask for neg counts to filter do normal or scoreing a waypt (dif scoring)
        print("I AM HAVINGGGGGGG")

        drone_x = self.latest_pos.x
        drone_y = self.latest_pos.y
        #TODO condiitonal for cur to go
        if self.cur_to_go.shape[0] != 0:
            dist_x = drone_x - self.cur_to_go[0]
            dist_y = drone_y - self.cur_to_go[1]

            squared_sum = pow(dist_x, 2) + pow(dist_y, 2)

            distance = math.sqrt(squared_sum)
            print("D: ", abs(distance))
        else:
            print("I AM HAVING HAVING")
        #Don't score until you have reached the last waypoint(unitl you having!)
        if self.scoring_flag:
            if persisted_array_all.size == 0:
                print("--------I AM NOT HAVING TRESS!--------")

            waypts_mask = persisted_array_all[:,3] < 0
            not_waypts = ~waypts_mask

            not_been_mask = persisted_array_all[:,4] == 0

            #-----Normal persistence scoring----
            trees = persisted_array_all[not_waypts & not_been_mask]

            persisted_counts = trees[:, 3]
            persisted_scores = (persisted_counts / (self.max_pers_counts)) * self.persisted_scores_weight + self.tree_const #Persisted score is based on what the count is divided by the maximum count (see self.max_pers_counts)


            #-----Waypt persistence scoring----
            waypts = persisted_array_all[waypts_mask & not_been_mask]

            # print("HERE is waypt mask: ", waypts_mask)
            # print("HERE is not been mask: ", not_been_mask)
            # print("HERE is waypt mask anded with not been mask: ", waypts_mask & not_been_mask)
            # print("waypts: ", waypts, "\n")

            #Since we filter ones alr been to, can just pick first one (and will always want to go sequentially)
            waypt = waypts[0] 
            self.curwaypt = waypt

            waypt_persistence_score = self.max_pers_counts * self.per_waypt_weight
            persisted_scores = np.append(persisted_scores,waypt_persistence_score) #Adding the waypoint maxed per score at bottom, so know its a waypt


            if self.is_odom:
                norms_score = np.zeros(0)
                drone_x = self.latest_pos.x
                drone_y = self.latest_pos.y

                print("HERE is tree.size: ", trees.size)

                self.trees = trees

                if trees.size != 0:
                    x_dist = trees[:, 0] - drone_x
                    y_dist = trees[:, 1] - drone_y
                    xy_dist = np.column_stack((x_dist, y_dist))

                    wx_dist = waypt[ 0] - drone_x
                    wy_dist = waypt[1] - drone_y
                    wxy_dist = np.column_stack((wx_dist, wy_dist))

                    all_dist = np.vstack((xy_dist, wxy_dist))

                    #Make a seperrate "all_norms" and use this to the the normalize. then append to the reg norms like we usually do
                    norms = np.linalg.norm(all_dist, axis = 1)

                    # print("norms BEFORE FILTER: ",  norms)
                    # nearby_norms = norms <= (POINT_SPACING* self.keep_circle)
                    # norms[nearby_norms] = 100000
                    # print("norms AFTER FILTER: ",  norms)

                    #-----Normal dist scoring----
                    norms_score = (norms/np.max(norms)) * self.norms_scores_weight #Dist score is normalized to the max distance. Smaller dist score is betteer
                    # print("HER is norm_score: ", norms_score)


                    #-----Waypt dist scoring----
                    # wnorm = np.linalg.norm(wxy_dist, axis = 1)
                    # wnorms_score = - (wnorm/np.max(norms)) * self.norm_waypt_weight
                    # norms_score = np.append(norms_score, wnorms_score)
                    #choose the last one, and divide by self.normscorewwitght and then mult by -1 and self.wnorms wegith
                    norms_score[norms_score.shape[0] -1] = (norms_score[norms_score.shape[0] -1] / self.norms_scores_weight )* -1 * self.norm_waypt_weight

                else:
                    norms_score = [1]



                #Assesment, whichever linear combination is highest
                # print("\n-------------------------------------------------------------------------------") 
                # print("HERE is per scores: ", persisted_scores)
                # print("HERE is norm scores: ", norms_score)

                self.glo_per_score = persisted_scores
                self.glo_norm_score = norms_score

                self.assesment = persisted_scores - norms_score
                # print("HERE is asses: ", self.assesment)
                # print("---------------------------------------------------------------------------------\n") 

                #Finding the index of the max_score, this will be the tree we go to
                self.max_index = np.argmax(self.assesment)

                #TODO Add conditional see if max_index == last row (shape[0]) --> means a waypt!
                # print("HERE is max_indx: ", self.max_index)
                if self.max_index == self.assesment.shape[0]-1:
                    print("HERE IS Where to go for a WAYPOINT: ", waypt[0:2])
                    self.cur_to_go = waypt[0:2] #to_go is NOT quantized, since it is an actual place to go to.
                else:
                    print("HERE IS Where to go for a TREE: ", trees[self.max_index, 0:2])
                    self.cur_to_go = trees[self.max_index, 0:2] #to_go is NOT quantized, since it is an actual place to go to.
                print("I AM CUR TO GO UPDATE IN SCORING:", self.cur_to_go)
                #False after you set self.cur_to_go (you are not having!)
                self.scoring_flag = False
                # self.to_go(self.cur_to_go)
         
    def to_go(self, to_go):
        """!@brief Publishes a dot for the centriod to go to, as well as a position msg for SUPER
            @param to_go [in] The (x, y) target position, taken from self.cur_to_go (a tree centroid or waypoint chosen by scoring_func)
            @details Notice that to_go is not quantized. We want maximum precision to avoid collision.
            @see doggy_timer @see scoring_func"""
        print("---I GONNA HAVING TO GO---", self.cur_to_go)
        
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

    def dist_to_goal(self, cur_to_go):
        """!@brief Calculates the distance from the current odometry position to the place to go to.
            @param cur_to_go [in] The (x, y) target currently being pursued (self.cur_to_go)
            @details When within self.goal_tol of the target, marks it visited: sets the corresponding row's 'been' column to 1 in self.all_persisted_array, records the quantized target in self.qbeen so hasbeenhere/persist_bookkeeping treat it as already visited, and sets self.scoring_flag so scoring_func picks the next target.
            @note This is the point in the overall finite state machine where, on reaching a tree, the drone would transition into the ascension state (not yet implemented here) to do close-proximity tree scanning and scaling.
            @see goal_timer"""
        #Quantizing the centroid to go to, to allow for putting this in self.qbeen
        # print("HERE is latest pos in DIST_TO_GOAL: ", self.latest_pos)
        qto_go = np.floor(cur_to_go / self.tol) * self.tol

        drone_x = self.latest_pos.x
        drone_y = self.latest_pos.y

        dist_x = drone_x - cur_to_go[0]
        dist_y = drone_y - cur_to_go[1]

        squared_sum = pow(dist_x, 2) + pow(dist_y, 2)

        distance = math.sqrt(squared_sum)

        #If the to_go has been reached, then add qto_go 
        # print("D: ", distance)
        if distance < self.goal_tol:

            # self.to_go(cur_to_go)

            print("waypt reached")
            self.scoring_flag = True
            #Flip the been col value to 1 for the centriod we went to
            #TODO where is togo in self.all_ersiste... quantize it first?
            qall_persisted_array = np.floor(self.all_persisted_array / self.tol) * self.tol

            #.all(axis = 1) makes this a row-wise match: BOTH x and y must equal qto_go.
            #Without it the comparison is element-wise, so any row merely sharing an x OR a y gets flagged.
            row = np.where((qall_persisted_array[:,0:2] == qto_go).all(axis=1))[0]

            # print("self.all_persisted_array[row,3] : ::: : :: : :", self.all_persisted_array[row,3])
            # if self.all_persisted_array[row,3].all(axis = 1) < 0:
            #     print("------GOING TO WAYPOINT-----")
            #     self.waypoint_index += 1

            self.all_persisted_array[row, 4] = 1.0

            if self.qbeen.size > 0:
                # print("I AM NOT HAVING: ", self.qbeen)
                self.qbeen = np.vstack((self.qbeen, qto_go))

            else:
                # print("I AM HAVING self.qbeen: ", self.qbeen)
                self.qbeen = qto_go

            # self.is_odom = False #Set back to false, so can do this func until have odom data





        #TODO pass in the persisted numpy array. Then save the normalized counts as persisted scores. Then compute the distance from where rn (from odom) to the centroid of mid z-slice.

    def make_lines(self):
        """!@brief Builds RVIZ line segments along each persisted tree's stored bearing angle
            @details For every non-waypoint row of self.all_persisted_array(with [:,3] < 0), constructs a line of 50 points spanning [-self.backlen, self.linelen] along the (cos(angle), sin(angle)) direction, centered at the tree's (x, y); appended to self.hlines for hesp_line_publ to publish.
            @note Distinct from PCA_make_lines: this uses the persisted tree's stored view-bearing angle, not a PCA Z eigenvector. not published here, just building self.hlines
            @see hasbeenhere @see hesp_line_publ"""
        waypts_mask = self.all_persisted_array[:,3] < 0
        not_waypts = ~waypts_mask
        trees = self.all_persisted_array[not_waypts]

        vec = np.column_stack((np.cos(trees[:,2]), np.sin(trees[:,2])))

        if vec.shape != (0,):
            length = self.linelen # i thiiink thats what ths is
            line = np.linspace(-self.backlen,length,50)[:,np.newaxis]


            for i in range(vec.shape[0]):
                self.hlines.append(line * vec[i] + trees[i,0:2])

#------------------------------------------------------------------------------------------------Goal and Watchdog Timer----------------------------------------------------------------------------------
    def goal_timer(self, event):
        """!@brief Timer callback that checks progress toward the current goal on every tick.
            @param event [in] TimerEvent automatically supplied by rospy.Timer
            @see dist_to_goal"""
        # with self.lock:
        if self.cur_to_go.size != 0:
            self.dist_to_goal(self.cur_to_go)

    def doggy_timer(self, event):
        """!@brief Watchdog function,periodically checks if the drone is moving towards the current waypoint, added so we wouldn't need to spam super
            @details If the drone is not moving towards the waypoint, it republishes the current waypoint to the /super/goal topic. Registered as a rospy.Timer callback in __init__, not called directly by run().
            @see to_go"""
        # print("HERE is latest pos in WATCHDOG: ", self.latest_pos)
        
        #Wating until 5 secs of odom data, to see if drone moving
        if self.cur_to_go.size != 0:
            if len(self.odom_list):
                delta_x = self.odom_list[-1].x - self.odom_list[0].x #last odom x - first odom x, to see if drone moved in x direction
                delta_y = self.odom_list[-1].y - self.odom_list[0].y
                delta_z = self.odom_list[-1].z - self.odom_list[0].z

                # print("Delta x: ", delta_x)
                # print("Delta y: ", delta_y)
                # print("Delta z: ", delta_z)

                if abs(delta_x) < 0.5 and abs(delta_y) < 0.5 and abs(delta_z) < 0.5: #and abs(delta_z) < 0.5: #Checking if odom x,y,z not changed much, if so then publish goal again so drone move
                    print("---I AM HAVING DOGGY---")
                    self.to_go(self.cur_to_go)
                self.odom_list.clear()
            
#---------------------------------------------------------------------------------------------Publishing Thread--------------------------------------------------------------------------------
    
    def all_vpancakes_publ(self, header):
        """!@brief Publishes all vertically-stacked pancake cluster points (self.all_vpancakes) as a PointCloud2 for rviz debugging.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see publ"""
        if self.all_vpancakes.any() != None:
            #BUG uncomment and fix the can't concatinate error
            # print("HERE IS publish_list: ", self.publish_list)
            # stacked_pub_list = np.vstack(self.publish_list)
            cluster_cloud = make_pointcloud2_xyz32(header, self.all_vpancakes)
            self.pub_cloud.publish(cluster_cloud)

    def pub_line_list_publ(self, header):
        """!@brief Publishes the PCA lines of clusters classified as vertical (candidate trees) as a PointCloud2.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see PCA_make_lines @see publ"""
        # print("Before if state here the pub linke ist: ", self.pub_line_list)
        if len(self.pub_line_list):
            # print("HERE is self.pub_line_list: ", self.pub_line_list)
            line_stacked = np.vstack(self.pub_line_list)
            line_cloud = make_pointcloud2_xyz32(header, line_stacked)
            self.pub_line.publish(line_cloud)

    def pub_not_line_list_publ(self, header):
        """!@brief Publishes the PCA lines of clusters classified as non-vertical (rejected, not trees) as a PointCloud2.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see PCA_make_lines @see publ"""
        if len(self.pub_not_line_list):
            not_line_stacked = np.vstack(self.pub_not_line_list)
            not_line_cloud = make_pointcloud2_xyz32(header, not_line_stacked)
            self.pub_not_line.publish(not_line_cloud)

    def pub_persisted_array_publ(self, header):
        """!@brief Publishes the current persisted tree/waypoint positions (self.pub_persisted_array) as a PointCloud2 for rviz.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see publ"""
        if self.pub_persisted_array.shape != (0,):
            persisted_dots = make_pointcloud2_xyz32(header, self.pub_persisted_array)
            self.pub_persisted.publish(persisted_dots)

    def intersect_pub_array_publ(self, header):
        """!@brief Publishes midpoints between paired centroids that hasbeenhere() flagged as intersecting/near-parallel sightlines, for debugging duplicate-tree detection.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see hasbeenhere @see publ"""
        if self.intersect_pub_array.shape != (0,):
            inter_dots = make_pointcloud2_xyz32(header, self.intersect_pub_array)
            self.pub_intersect.publish(inter_dots)

    def hesp_line_publ(self,header):
        """!@brief Publishes the persisted-tree bearing-direction lines (self.hlines) as a PointCloud2 for rviz.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see make_lines @see publ"""
        if len(self.hlines):
            # print("HERE is hlines: ", self.hlines)
            line_stacked = np.vstack(self.hlines)
            z_ones = np.ones((line_stacked.shape[0],1)) * 0.67
            line_stacked_with_z = np.column_stack((line_stacked,z_ones))
            hline_cloud = make_pointcloud2_xyz32(header, line_stacked_with_z)
            self.pub_hespline.publish(hline_cloud)

    def eig_cond_text_publ(self,header):
        """!@brief Publishes a floating rviz text marker per mid z-slice cluster, showing its RMS/elongation and PCA eigenvector diagnostics for debugging.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @see centroid_finder @see kd_tree_PCA @see publ"""
        if len(self.text_dict) > 0:

            marker_array = MarkerArray()

                

            for cluster in enumerate(self.text_dict):
                    marker = Marker()
                    marker.header = header
                    marker.ns = "text_messages"
                    marker.id = cluster[0]  # Unique ID per text string
                    marker.type = Marker.TEXT_VIEW_FACING
                    marker.action = Marker.ADD


                    clus_num = cluster[1] #API have to do, 0 is index
                    # Position of the text in 3D space
                    marker.pose.position.x = self.text_dict[clus_num]["xmean"]
                    marker.pose.position.y = self.text_dict[clus_num]["ymean"]
                    marker.pose.position.z = 10
                    marker.pose.orientation.w = 1.0
                    
                    # Text scale/size (Z controls height of capital letters)
                    marker.scale.z = 0.15

                    # Text color
                    marker.color.r = 0
                    marker.color.g = 0
                    marker.color.b = 1.0
                    marker.color.a = 1.0


                    hor_rms = round(self.text_dict[clus_num]["hor_rms"], 4)
                    ver_rms = round(self.text_dict[clus_num]["ver_rms"], 4)
                    elong = round(self.text_dict[clus_num]["elongation_num"], 4)

                    x_eig = np.round(self.text_dict[clus_num]["x_eig"], decimals=4)
                    x_index = self.text_dict[clus_num]["x_index"]
                    y_eig = np.round(self.text_dict[clus_num]["y_eig"], decimals=4)
                    y_index = round(self.text_dict[clus_num]["y_index"], 4)
                    z_eig = self.text_dict[clus_num]["z_eig"]
                    z_index = np.round(self.text_dict[clus_num]["z_index"], decimals=4)
                    
                    marker.text = f"hor_rms: {hor_rms}, ver_rms: {ver_rms}, elongation: {elong}, \nx_eig: {x_eig}, x_index: {x_index}, \ny_eig: {y_eig}, y_index: {y_index}, \nz_eig: {z_eig}, z_index: {z_index}"
                    marker.lifetime = rospy.Duration(0.1)  # Refresh duration
                    
                    marker_array.markers.append(marker) 

            self.pub_text.publish(marker_array)


    def assesment_score_text_publ(self, header):
        """!@brief Publishes a floating rviz text marker per scored candidate, showing its combined assessment score and persistence/distance components, for debugging scoring_func's target choice.
            @param header [in] std_msgs Header (frame + timestamp) shared by all publ() outputs this cycle
            @bug The loop only runs for index in range(self.assesment.shape[0]-1), so index never reaches self.assesment.shape[0] (the `else` branch compares against, meant to catch the waypoint slot per scoring_func's convention); the waypoint candidate is therefore never plotted, and the last tree candidate is skipped.
            @see scoring_func @see publ"""
        if len(self.text_dict) > 0:
            smarker_array = MarkerArray()
            for index in range(self.assesment.shape[0]-1):
                # print('HERE s index: ', index)
                marker = Marker()
                marker.header = header
                marker.ns = "text_messages"
                marker.id = index  # Unique ID per text string
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                if index != self.assesment.shape[0]:
                    # print("HERE is shape of trees: ", self.trees)
                    peni = self.trees[index, 0]
                    marker.pose.position.x = peni
                    marker.pose.position.y = self.trees[index, 1]

                else:
                    marker.pose.position.x = self.curwaypt[0]
                    marker.pose.position.y = self.curwaypt[1]

                marker.pose.position.z = 10
                marker.pose.orientation.w = 1.0

                # Text scale/size (Z controls height of capital letters)
                marker.scale.z = 0.7

                # Text color
                marker.color.r = 0
                marker.color.g = 0
                marker.color.b = 1.0
                marker.color.a = 1.0

                marker.text = f"Asses: {round(self.assesment[index],2)},\nPer: {round(self.glo_norm_score[index],2)},\nNorm: {round(self.glo_norm_score[index],2)}"
                marker.lifetime = rospy.Duration(1.5)  # Refresh duration
                smarker_array.markers.append(marker)

            self.pub_text.publish(smarker_array)
    
    def publ(self):
        """!@brief Publishes all rviz debug topics for the current cycle once kd_tree_PCA has produced fresh results.
            @details Builds one shared Header and dispatches to each *_publ helper.
            @see all_vpancakes_publ @see pub_line_list_publ @see pub_not_line_list_publ @see pub_persisted_array_publ @see intersect_pub_array_publ @see hesp_line_publ @see assesment_score_text_publ"""
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        self.all_vpancakes_publ(header)
        self.pub_line_list_publ(header)
        self.pub_not_line_list_publ(header)
        self.pub_persisted_array_publ(header)
        self.intersect_pub_array_publ(header)
        self.hesp_line_publ(header)
        self.assesment_score_text_publ(header)


def main():
    rospy.init_node("TreeFinder") #Make the node

    TreeFinder().run()

if __name__ =="__main__":
    main()