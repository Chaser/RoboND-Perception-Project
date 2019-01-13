# Project: Perception Pick & Place

## Engineer: Chase Johnson

[//]: # (Image References)
[image1]: ./img/project_intro.png
[image2]: ./img/rgbd_capture.png
[image3]: ./img/pcd_vox_downsample.png
[image4]: ./img/filter_vox_downsample.png
[image5]: ./img/filter_passthrough.png
[image6]: ./img/filter_inlier.png
[image7]: ./img/filter_outlier.png
[image8]: ./img/dbscan_example.png
[image9]: ./img/cluster_segmentation.png
[image10]: ./img/capture_feature_example.png
[image11]: ./img/svm_train_confusion_matrix_10_samples_64_bin_linear_nfolds_5.png
[image12]: ./img/svm_train_normalized_confusion_matrix_10_samples_64_bin_linear_nfolds_5.png
[image13]: ./img/svm_train_confusion_matrix_20_samples_64_bin_linear_nfolds_5.png
[image14]: ./img/svm_train_normalized_confusion_matrix_20_samples_64_bin_linear_nfolds_5.png
[image15]: ./img/svm_train_confusion_matrix_10_samples_128_bin_linear_nfolds_5.png
[image16]: ./img/svm_train_normalized_confusion_matrix_10_samples_128_bin_linear_nfolds_5.png
[image17]: ./img/classification_result_world_1.png
[image18]: ./img/classification_result_world_2.png
[image19]: ./img/classification_result_world_3.png

---

**Aim:**  The aim of the `Perception Pick & Place` project is to create 3D perception pipeline for a PR2 robot utilziing an RGB-D camera. The perception pipline allows for capturing sensor data to point cloud data (PCD), to filter, isolate and detect objects.

![alt text][image1]


This project is inspired by the [Amazon Robotics Challenge](https://www.amazonrobotics.com/site/binaries/content/assets/amazonrobotics/arc/2017-amazon-robotics-challenge-rules-v3.pdf)


# 3D Image Acquisition
Before doing anything we must acquire the point cloud data `PCD` from the RGBD sensor. The sensor conviently establishes a ROS topic, specifically `/pr2/world/points` so this can be subscribed to as shown below:

```python
# ROS node initialization
rospy.init_node('clustering', anonymous=True)

# Create Subscribers
pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
```

To validate the subscription was working correctly the point cloud data aquired from the RGB-D sensor and passed to the callback was looped back to a test topic.

```python
# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    ## Convert ROS msg to PCL data ##
    pcl_data = ros_to_pcl(pcl_msg)
    # ...
    # TODO: Convert PCL data to ROS messages
    ros_cloud_outliers_filtered = pcl_to_ros(pcl_data)

    # TODO: Publish ROS messages
    pcl_outliers_filtered_pub.publish(ros_cloud_outliers_filtered)


if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_outliers_filtered_pub = rospy.Publisher("/pcl_outliers_filtered", PointCloud2, queue_size=1)
```

![alt text][image2]

# Point Cloud Filtering
RGB-D sensors provide a wealth of information, however the majority of the data is not useful for identifying the targets. As we learnt in the lessons.

`"Running computation on a full resolution point cloud can be slow and may not yield any improvement on results obtained using a more sparsely sampled point cloud"` 

In our case we will take advantage of three filtering techniques, specifically `Outlier Removal Filter`, `Voxel Grid Downsampling` and then a `Pass through filter`

## Outlier Removal Filter
Outliers create statistical noise and one such technica to remove such outlisers is by performing statistical analysis on neighbouring points.

```python
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
```

## Voxel Grid Downsampling
As indicated above, performing computations on dense point clouds can be slow and not provide any benfit. Downsampling is a method of quantizing the captured data to derive a point cloud that has fewer points. The density of the point clouds is by the `leaf` size which correlates to the sampling size.

```python
## Voxel Grid Filter ##
# Create a VoxelGrid filter object for our input point cloud
vox = pcl_data.make_voxel_grid_filter()
LEAF_SIZE = 0.01
# Set the voxel (or leaf) size
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
```

The result produces a less dense point cloud which will decrease computation time without reducing the quality of detection.

![alt text][image3]

![alt text][image4]

## Pass through Filtering
While vox filtering has reduced the density of the point cloud data that will be examined there is one other step we can perform and thats because we have prior knowledge of the scenario. Specifically we know that objects will exist on the table and we know where the size and location of the table. Therefore we can "crop" out irrelvant areas of the scene, creating a `region of interest`.

```python
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
```

![alt text][image5]


# RANSAC Plane Segmentation
The next stage of our preception pipeline is to remove the table, which will done with Random Sample Consensus or `RANSAC`. It is used to identify points in a dataset that belong to a particular model. It assumes that the data in a dataset is composed of inliers, which have a specific set of parameters and outliers that don't. 

In this particular case we have prior knowledge of the table, shape and size. By modelling the table as a plane it can be removed from the point cloud. 

```python
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
```

Below is the image of the inliers (Table) and the outliers (objects of interest)

![alt text][image6]

![alt text][image7]


# Clustering for Segmentation
Density-Based Spatial Clustering of Applications with Noise or `DBSCAN`. `DBSCAN` creates clusters by grouping data points by a threshold from the nearest neighbours (shown below). The algorithm is sometimes called `Euclidean Clustering` as the decision for a point to reside within a cluster is based ipon the "Euclidean distance" between that point and other cluster members.

![alt text][image8]

`DBSCAN` has the a numer of advantages over other clustering techniques (e.g `K-means`) as it can be used when there are a **unknown** number of clusters present. This makes ideal for the current project as we don't know how many objects will be present.

```python
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
```

By creating a random list of colours and assigning each to a cluster the following objects can be seen.

![alt text][image9]


# Object Recognition Training
Object recogntion is the ability to detect the objects by examining features. However, features have to be taught or trained to know what the expected values of the objects features are. 

Two particular features we will use to classify the objects we are examining are `colour histogram` and `surface normals`. 

## Capturing Object Features
Capturing the objects features is performed by spawning (`spawn_model`) a number of a random models from different angles while extracting the `colour` and `surface normals` and labelling each set (code below)

```python
# Extract histogram features
chists = compute_color_histograms(sample_cloud, using_hsv=True, bin_size=hist_bin)
normals = get_normals(sample_cloud)
nhists = compute_normal_histograms(normals, bin_size=hist_bin)
feature = np.concatenate((chists, nhists))
labeled_features.append([feature, model_name])
```

![alt text][image10]

Example of randomized model generation

Two changes were made from the default settings, firstly `RGB` was translated to `HSV` due to it being less senstive to lighting conditions. Also the number of randomized model conditions was increased from 5 to 10 but also experimented with 20. The `HSV` bin size was also varied from `64` to `128` to examine changes.

## Training the SVM Model
Supported Vector Machine (SVM), a supervised machine learning algorithm by applying an iterative method where each in the training set is characterized by a feature vector and a label.

In this case, each item had two features, `colour` based on `HSV` and `surface` noramals. `train_svm.py` was used to exeperiment with a number of parameters (`svm-data.md`) 

### SVM Training Results
Out of the box using history grams with 64 bins the SVM revealed a 94.5% accuracy

![alt text][image11]![alt text][image12]

Increasing the sample size from `10` to `20` saw a drop to 93.3%

![alt text][image13]![alt text][image14]

Increasing the histograms bin size from `64` to `128` saw an increas from 94.5% to 97.7%

![alt text][image15]![alt text][image16]


Seeing the model improve in some respects and degrade in others was interesting.

# Object Recognition Results

The classification and prediction pipeline stage is below:

```python
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
```

## Test World 1
Test one was an easy experiment with 100% (3/3) objects being detected.

![alt text][image17]

## Test World 2
The second world had a number of objects that looked similar however the detection algorithm was still able to detect 100% (5/5)

![alt text][image18]

## Test World 3


![alt text][image19]


# Pick and Place Determination

The final component to the project, was to pass the detected objects to the `pr2_mover` function for the `pick_place_routine` of the robot. The results of this were to be sorted in a `yaml` file which provides details about the object and its associated `pick` and `place` locations.

An example of the `yaml` output file.

```yaml
object_list:
- arm_name: left
  object_name: book
  pick_pose:
    orientation:
      w: 0.0
      x: 0.0
      y: 0.0
      z: 0.0
    position:
      x: 0.49258291721343994
      y: 0.08352160453796387
      z: 0.7266173362731934
  place_pose:
    orientation:
      w: 0.0
      x: 0.0
      y: 0.0
      z: 0.0
    position:
      x: 0
      y: 0.71
      z: 0.605
  test_scene_num: 1
```

# Final Result and Improvements.

The project succesfully achieved the results by identifying the objects within the test worlds. The next steps would be to continue to experiment with the SVM traning, specifically the Radial Basis Function `RBF` kernel. Also to examine the collision map and review the current objection recognition in the complex world.