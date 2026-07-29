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



#TODO We want to do two things A) find poss trees B) sep class at least, goes up to poss tree, and estimates radius, and reonsiders dist from tree for cam, and reconsiders tree placements
#b would happen before and while going towards tree, 
#When state 3 happens (assecnetion), that's when we know for certain the rad of the tree, and can accurately place waypt,next waypoint can b determined

class TreeFinder:
    def __init__(self): 
        self.lock = threading.Lock()

        #Publish beacon/waypoint for found trees
        self.pub_beacon = rospy.Publisher("/Beacons", PointCloud2, queue_size = 10)

        self.pub_cloud = rospy.Publisher("/Z_CLUSTERS", PointCloud2, queue_size = 10)

        self.pub_line = rospy.Publisher("LINES", PointCloud2, queue_size = 10)

        self.pub_text = rospy.Publisher('/TEXT', MarkerArray, queue_size=10)

        #Sub to cum cloud
        self.sub = rospy.Subscriber("/Cum_Cloud", PointCloud2, self.cloud_cb, queue_size = 10)
        self.cloud_section_one = None
        self.cloud_section_two = None

        #Timer
        rospy.Timer(rospy.Duration(0.1), self.on_timer)  # 10 Hz

        self.latest_cloud = None
        
        self.pancake_stacks = 7
        self.pancake_start = round(5 * 0.067, 5)  #5 before
        self.pancake_gap = round(1 * 0.067, 5) #TODO This has to be small for new alg
        self.pancake_thickness = round(1 * 0.067, 5)
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
        self.processed_cloud_list = []

        self.publish_list = deque(maxlen =self.pancake_stacks*1)

        #------New alg---------
        self.mid_z_dict = defaultdict(dict) #Stores relevant info for mid z slice clusters: cluster, num pts in cluster, centriod
        self.mid_n_clusters = 0
        self.clustered_cloud_list = []
        self.all_vpancakes = None
        self.pub_line_list= []
        self.leaf_size = 80

        self.text_dict = defaultdict(dict)


        #TODO OLD STUFF: Tree Confirmation Parameters

        #Step 1: Is Line
        self.hor_rms_threshold = 0.8 #Measure of how spread horizontally
        self.elongation_num_threshold = 1 #ranges 0-1, higher is more "circular"
        


    def run(self):
        rospy.spin()

    

    def cloud_cb(self, msg):
        self.latest_cloud = self.cloud_to_xyz(msg)
        self.processed_cloud_list.clear()

        for i in range(self.pancake_stacks):
            cur_mid_height = self.pancake_start + (self.pancake_gap * i)  #Middle height of cur pancake looking at
            processed_cloud = self.cut_cloud(self.latest_cloud, cur_mid_height)
            self.processed_cloud_list.append(processed_cloud)

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
        # self.publish_list = np.append(self.publish_list,cut_cloud[first],axis = 0)
        self.publish_list.append(cut_cloud[first])

        # print("publish list size: asdfasdfasdfasdf",np.shape(self.publish_list))
        # print("cut cloud shape asdfasfasfasdfasdf " , np.shape(cut_cloud[first]))

        #Return cut_cloud with indices that only include uniqe x,y's
        return cut_cloud[first,0:2] 

    def on_timer(self,event):
        i = 1
        with self.lock:
            if len(self.processed_cloud_list):
                mid_num = round(self.pancake_stacks / 2)
                for pancake_num in range(self.pancake_stacks):
                    is_mid = False
                    # print("HERE is mid_num: ", mid_num)

                    if i == mid_num: #Checking if at the mid z-slice
                        is_mid = True

                    self.mid_z_dict.clear()#Clear dict for mid slice, before poulate again with new clustering
                    e_array = self.centroid_finder(pancake_num, is_mid) #TODO pass another param here ditate whether at mid cluster, so execute dif logis in centriod_finder

                    if e_array is not None:
                        # not_zero_mask = [e_array != (0,0,0,0,0,0)]
                        # print("EARRAY: ", e_array)
                        # print("EARRAY shape: ", e_array.shape)

                        not_zero_mask = np.any(e_array != 0, axis =1)
                        # print("HERE IS not zero mask: ", not_zero_mask)
                        cleaned_e_array = e_array[not_zero_mask]

                        self.cand_trees[pancake_num] = cleaned_e_array

                    i += 1

                
                    

                        #TODO somehwere in on_timer, need to call func for kd tree and PCA
            # print("HERE is self.cand_tree: ", self.cand_trees)
            # print("HERE is self.cand_tree shape: ", self.cand_trees.shape
            self.publ()
            self.pub_line_list.clear()
            self.text_dict.clear()
        #TODO  call cluster categorizing steps 2-4 funcs here

    def centroid_finder(self, which, is_mid):
        # print("HERE is which inside of centriod findeer: ", which)
        # print("proc cloud list length::::::::::::: " ,len(self.processed_cloud_list))
        count = 0
        if len(self.processed_cloud_list) > which:
            processed_cloud = self.processed_cloud_list[which]
            if len(processed_cloud):
                labels, n_clusters = self.clustering(processed_cloud, which)

            # self.centroid_list = np.zeros((n_clusters-1,3))
            if n_clusters > 0:
                e_array = np.zeros((n_clusters-1, 6))
                self.clustered_cloud_list = self.xy[labels != -1]
                # print("HERE is clustered_cloud_list: ", self.clustered_cloud_list)

                for clustnum in range(n_clusters-1):
                    curr_clust = self.xy[labels == clustnum]
                    xmean = np.mean(curr_clust[:,0])
                    ymean = np.mean(curr_clust[:,1])

                    #Getting relevant cluster info from eigen func
                    hor_rms, ver_rms, elongation_num = self.eigen(curr_clust, xmean, ymean)

                    clust_name = f"Cluster {count}"

                    #Putting hor_rms, xmean, and ymean in a dict dynamically to pub text
                    self.text_dict[clust_name]["hor_rms"] = hor_rms
                    self.text_dict[clust_name]["ver_rms"] = ver_rms
                    self.text_dict[clust_name]["xmean"] = xmean
                    self.text_dict[clust_name]["ymean"] = ymean
                    self.text_dict[clust_name]["elongation_num"] = elongation_num

                    #Populate mid_z_dict if at mid z-sclie
                    if is_mid and not self.is_line(hor_rms, elongation_num):
                        # print("HERE is curr_clsut: ", curr_clust)
                        num_pts = curr_clust.shape[0]

                        #TODO replace this call with the filtering func
                        # print("HERE is num of ptS: ", num_pts)
                        self.mid_z_dict[clust_name]["num_pts"] = num_pts
                        self.mid_z_dict[clust_name]["xmean"] = xmean
                        self.mid_z_dict[clust_name]["ymean"] = ymean

                        count += 1
                    



                        # print("HERE is mid_z_dict: ", self.mid_z_dict)

                            #Populating alr instantiated numpy array in mem. This array holds cluster info for all clusters in a z "pancake" slice
                            # e_array[clustnum] = [hor_rms, ver_rms, elongation_num, which,xmean,ymean]

                        # print("HER is e_array shape: ", e_array.shape)
                    #TODO if else statement for returned value for is big func ARTIFACT???

                #TODO
                # print("HERE is count: ", count)
                self.kd_tree_PCA(count) #Call #TODO filtering func, after for loop so dict is fully populated
                return e_array
            return None

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
    



    #------Tree Confirmation Step Func-----
    def eigen(self, curr_clust, xmean, ymean):
            normalized = curr_clust - np.array([xmean,ymean]) #To do it at origin
            #  print("HERE is normalized shape: ", normalized.shape)
    
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
        if hor_rms >= self.hor_rms_threshold: #or elongation_num <= self.elongation_num_threshold:
            print("HERE is hor_rms fFOR LINE ", hor_rms) 
            return True

        return False
         

    #--------New alg funcs--------
    #TODO KDtree. This will access elements from mid-z-slice dict. Outputs numpy arrays for the 3D assoicated clusters for potential trees.
    #Use processed cloud llist, to access all the "pancakes"
    def kd_tree_PCA(self, n_clusters):
        if len(self.clustered_cloud_list):
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())
            # print("HERE is type of procecllist: ", self.processed_cloud_list.type())
            all_pancakes = []
            for i in range(self.pancake_stacks):
                curr_z = self.pancake_start + (self.pancake_gap * i)
                num_rows = np.shape(self.clustered_cloud_list)[0] 
                z_array = np.ones((num_rows,1)) * curr_z
                # print("z_array: ", z_array)
                # print("z_array shape: ", z_array.shape)
                all_pancakes.append(np.append(self.clustered_cloud_list,z_array,axis = 1))
                # TODO can optimize heavily if you just append xyz  of fullxyz[label boolean mask] instead of saving lists of x, y, then vstacking , do late r later

                # print("HERE is all_panacakes: ", all_pancakes)

            #V stacks all pancakes verically, to give KDtree all points from all pancakes

            # print("HERE is all_pancakes: ", all_pancakes)
            self.all_vpancakes = np.vstack(all_pancakes)
            

            if n_clusters > 0 and len(self.mid_z_dict):
                for i in range(n_clusters-1):
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

                    # print("HERE is max_z_index: ", max_z_index)
                    # print("HERE is z_eig: ", z_eig, "\n")
                    self.PCA_make_lines(z_eig, centriod)

                    # for eig_vec in fitted_comps:
                    #     if eig_vec[2] != 
                    #         print("EIG VEC: ", eig_vec)
                    #         self.PCA_make_lines(eig_vec,centriod)

                    # self.PCA_zaxis_list.append(fitted_comps[0] + centriod)
                    # print("HERE is pca axis's: ", fitted_comps)

    def PCA_make_lines(self, z_axis,centroid):
        z_axis = abs(z_axis)
        z_basis = z_axis / np.linalg.norm(z_axis)

        length = 10
        line = np.linspace(0,length,20)[:,np.newaxis]
        # print("line shape: ", line.shape)
        # print("line: ", line)
        # print("line z_basis: ", z_basis)

        curr_line = line * z_basis + centroid
        # print("HERE is curr_line: ", curr_line)
        self.pub_line_list.append(curr_line)
        # print("HERE is pub line list")



  

    #TODO Filtering func. This will call PCA func, take the outputed numpy array (either reutnr of make global idk yet), and pass into PCA func. This willl then
    #take the poential trees PCA said aren't trees, and filter the cand_trees (or make a new list) accordingly

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

    def publ(self):#TODO publish the PCA z-axis???
        header = std_msgs.msg.Header(frame_id = "camera_init", stamp = rospy.Time.now())
        if self.all_vpancakes.any() != None:
            #TODO uncomment and fix the can't concatinate error
            # print("HERE IS publish_list: ", self.publish_list)
            # stacked_pub_list = np.vstack(self.publish_list)
            cluster_cloud = self.make_pointcloud2_xyz32(header, self.all_vpancakes)
            self.pub_cloud.publish(cluster_cloud)
        # print("Before if state here the pub linke ist: ", self.pub_line_list)
        if len(self.pub_line_list):
            # print("HERE is self.pub_line_list: ", self.pub_line_list)
            line_stacked = np.vstack(self.pub_line_list)
            line_cloud = self.make_pointcloud2_xyz32(header, line_stacked)
            self.pub_line.publish(line_cloud)

        #TODO check if the text dict is len, then publish
        if len(self.text_dict):
            marker_array = MarkerArray()

            for cluster in enumerate(self.text_dict):
                marker = Marker()
                marker.header = header
                marker.ns = "text_messages"
                marker.id = cluster[0]  # Unique ID per text string
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD

                # print("HERE is text dict: ", self.text_dict)
                # print("HERE is cluster var: ", cluster[1])
                clus_num = cluster[1]
                # Position of the text in 3D space
                marker.pose.position.x = self.text_dict[clus_num]["xmean"]
                marker.pose.position.y = self.text_dict[clus_num]["ymean"]
                marker.pose.position.z = 10
                marker.pose.orientation.w = 1.0
                
                # Text scale/size (Z controls height of capital letters)
                marker.scale.z = 0.3
                
                # Text color
                marker.color.r = 0
                marker.color.g = 0
                marker.color.b = 0
                marker.color.a = 1.0

                hor_rms = round(self.text_dict[clus_num]["hor_rms"], 4)
                ver_rms = round(self.text_dict[clus_num]["ver_rms"], 4)
                elong = round(self.text_dict[clus_num]["elongation_num"], 4)
                
                marker.text = f"hor_rms: {hor_rms}, \nver_rms: {ver_rms}, \nelongation: {elong}"
                marker.lifetime = rospy.Duration(0.1)  # Refresh duration
                
                marker_array.markers.append(marker)

            self.pub_text.publish(marker_array)

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