#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    
    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    # Create a filter object: 
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(5)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    outliers_filtered = outlier_filter.filter()
   
    # Voxel Grid Downsampling
    # Create a VoxelGrid filter object for our input point cloud
    vox = outliers_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.01
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    downsampled = vox.filter()
    filename = 'voxel_downsampled.pcd'
    #pcl.save(downsampled, filename)

    # PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = downsampled.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.60
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    # Call the filter function to obtain the resultant
    cloud_filtered = passthrough.filter()
    # Also limit the y axis to avoid the side bins
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.42
    axis_max = +0.42
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    filename = 'passthrough_filter.pcd'
    #pcl.save(cloud_filtered, filename)

    # RANSAC plane segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    # Extract inliers (tabletop)
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    # Extract outliers (objects)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
 
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)  # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(2500)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

  
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # Convert PCL data to ROS messages
    ros_cloud_outliers_filtered = pcl_to_ros(outliers_filtered)
    ros_cloud_downsampled =  pcl_to_ros(downsampled)
    ros_cloud_filtered = pcl_to_ros(cloud_filtered)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_outliers_filtered_pub.publish(ros_cloud_outliers_filtered)
    pcl_downsampled_pub.publish(ros_cloud_downsampled)
    pcl_filtered_pub.publish(ros_cloud_filtered)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects_list = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        colour_hists = compute_color_histograms(ros_cluster, using_hsv=True, bin_size=64)
        normals = get_normals(ros_cluster)
        norm_hist = compute_normal_histograms(normals, bin_size=64)
        feature = np.concatenate((colour_hists, norm_hist))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects_list)


    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 3
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    yaml_dict_list = []
    yaml_file_name = "output_%i.yaml".format(test_scene_num)

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    try:
        # key = "name", value = "group"
        pick_object_group = {item['name']: item['group'] for item in object_list_param}
        # key = "group", value = "position"
        group_position = {item['group']: item['position'] for item in dropbox_list_param}
        # key = "group", value = "name"
        group_arm = {item['group']: item['name'] for item in dropbox_list_param}
    except KeyError as ke:
        print("Invalid object, didnt have Key %s", ke)
        return
    
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for detected_obj in object_list:
        object_name = detected_obj.label
        object_group = pick_object_group.get(object_name)

        # Get the PointCloud for a given object and obtain it's centroid
        labels.append(detected_obj.label)
        points_arr = ros_to_pcl(detected_obj.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

        # Determine 'pick_pose' for the object
        pick_pose = Pose()
        pick_pose.position.x = centroids[0]
        pick_pose.position.y = centroids[1]
        pick_pose.position.z = centroids[2]

        # Create 'place_pose' for the object
        place_pose = Pose()
        place_pose.position.x = group_position.get(object_group)[0]
        place_pose.position.y = group_position.get(object_group)[1]
        place_pose.position.z = group_position.get(object_group)[2]

        # Assign the arm to be used for pick_place
        arm_name = group_arm.get(object_group)

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_outliers_filtered_pub = rospy.Publisher("/pcl_outliers_filtered", PointCloud2, queue_size=1)
    pcl_downsampled_pub = rospy.Publisher("/pcl_downsampled", PointCloud2, queue_size=1)
    pcl_filtered_pub = rospy.Publisher("/pcl_filtered", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model_filename = 'model_{0}_samples_{1}_bin_{2}_nfolds_{3}.sav'.format(10, 64, 'linear', '5')
    model = pickle.load(open(model_filename, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
