#!/usr/bin/env python3
import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
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

        #TODO publish beacon/waypoint for found trees

        #TODO sub to cum cloud
        #Timer
        rospy.Timer(rospy.Duration(.1), self.on_timer)  # 10 Hz


    def run(self):
        rospy.spin()
    #TODO ontimer functin, reads possible trees, does beacon publishing

    #TODO cb for getting cloud data, consider downsampling once more here, or doing 2d slices by callign func

    #TODO func to do 2d slice of cloud data, think optimizing, don't jsut do 1 slice, do 2-3 to know its a tree (ver. cyl ah shape ish)

    #TODO func to cluster data and find poss trees

def main():
    rospy.init_node("TreeFinder") #Make the node

    TreeFinder().run()

    # rospy.spin()

if __name__ =="__main__":
    main()