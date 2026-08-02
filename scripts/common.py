from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
"""!@file common.py
    @brief Contains common functions used in the perception package"""

def make_pointcloud2_xyz32(header, points):
    """!@brief Build an XYZ32 PointCloud2 message via one tobytes() instead of a Python
    per-point pack loop (which is what create_cloud_xyz32 does internally).
        @note bytes are little endian
        @param header The header for the PointCloud2 message
        @param points A numpy array of shape (N, 3) containing the XYZ coordinates
        @return A PointCloud2 message containing the XYZ coordinates in the data field"""
    
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

def cloud_to_xyz(msg):
    """!@brief Converts a PointCloud2 message to a numpy array of XYZ coordinates
        @details PointCloud2 message is a 1D byte buffer, with x, y, z, and a pad. x, y, z are float32 at offsets 0, 4, and 8 respectively. The pad contains additional info about the point cloud. Using np.frombuffer(), it is possible to read, not copy, the
        data into a structured numpy array with fields defined by a dtype matching the PointCloud2 message format.
        @return A numpy array of shape (N, 3) containing the XYZ coordinates of the points in the PointCloud2 message.
        @note The purpose of using byte bufferis to optimize for speed by avoiding nested loops and numpy's non-vectorized functions.
        @param msg The PointCloud2 message to convert."""
    
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