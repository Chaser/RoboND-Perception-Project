# Project: Perception Pick & Place

## Engineer: Chase Johnson

---
[//]: # (Image References)
[image1]: ./img/project_intro.png
[image2]: ./img/rgbd_capture.png
[image3]: ./img/pcd_vox_downsample.png
[image4]: ./img/filter_vox_downsample.png
[image5]: ./img/pcd_passthrough.png
[image6]: ./img/filter_passthrough.png
---


**Aim:**  The aim of the `Perception Pick & Place` project is to create 3D perception pipeline for a PR2 robot utilziing an RGB-D camera. The perception pipline allows for capturing sensor data to point cloud data (PCD), to filter, isolate and detect objects.

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
# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.60
axis_max = 0.9
passthrough.set_filter_limits(axis_min, axis_max)
# Also limit the y axis to avoid the side bins
filter_axis = 'y'
passthrough.set_filter_field_name(filter_axis)
axis_min = -0.42
axis_max = +0.42
passthrough.set_filter_limits(axis_min, axis_max)
```

![alt text][image5]

![alt text][image6]


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



