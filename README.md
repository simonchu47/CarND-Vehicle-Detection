# Vehicle Detection Project

This project is the fifth project of Udacity Self-driving Car Nanodegree. The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notCar.jpg
[image2]: ./output_images/features.jpg
[image3]: ./output_images/test1_scanning_window.jpg
[image4]: ./output_images/test6_scanning_window.jpg
[image5]: ./output_images/video_frames_heatmap.jpg
[image6]: ./output_images/output_labels.jpg
[image7]: ./output_images/output_bbox.jpg
[image8]: ./output_images/all_scanning_windows.jpg
[video1]: ./output_images/project_video_vehicle_detecting.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images

The code for this step is contained in lines 33 through 50 of the file called `useful_functions.py`. The `skimage.feature.hog` function is used to extract HOG features after the image is converted to the desired color space. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. The final choice of HOG parameters

I tried various combinations of parameters which include `orientations`of value 6, 9, 12, `pixels_per_cell`of value 4, 8, 16, and `cells_per_block` of value 1, 2, 4 in training the classifier with all the `vehicle` and `non-vehicle` images. The more `orientations` the higher test accuracy, but more time needed. The more `pixels_per_cell` the lower test accuracy, but less time needed. The less `cells_per_block` the lower test accuracy, but less time needed.

In addition, all the test accuracy are between 0.978 and 0.993, and I choosed `orientations`of value 9, `pixels_per_cell`of value 8 and `cells_per_block` of value 2, and the test accuracy of such combinations was 0.9885, without spatial and color histogram features.

#### 3. Training a classifier using the selected HOG features

I trained a linear SVM using the extracted features of the `vehicle` and `non-vehicle` images. The features include YUV spatial, color histogram and HOG of each channel features. I have tried to convert these images into other color spaces and extracted the features, and I found that YUV color space is the one with high test accuracy which is 0.9935 but not too much training time which is 26.23s. Those features are normalized wtih `sklearn.preprocessing.StandardScaler()` and randomly splited into training and test sets, which are 80% and 20% of all repectively, with `sklearn.model_selection.train_test_split()`. Those steps are in lines 122 through 134 of the file called `vehicle_detection.py`.

I also used `sklearn.grid_search.GridSearchCV()` function to tune the `C` parameter of the linear SVM. The best `C` parameter is 1.0.
The steps mentioned are in lines 192 through 209 of the file called `vehicle_detection.py`

### Sliding Window Search

#### 1. The sliding window searching

I decided to search bottom halve of the read-in image with two scales of sliding window:

The large scale window is 128x128 pixels scanning from y position 400 to 656. The window is searching for vehicles that are closer to us. And the middle scale window is 64x64 pixels scanning from y position 400 to 500, and it is searching for vehicles that are more far away. This step is in lines 218 through 251 of the file called `vehicle_detection.py`.

And each step to move on for the scanning window is 2 cells, which means 16 pixels if 8 pixels per cell is set. Therefore the overlap of the windows is 75%. Below is an example of all the scanning windows for an image. 
![alt text][image8]

#### 2. The working pipeline and optimization of the classifier

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

The classifier was optimized by `sklearn.grid_search.GridSearchCV()`, with the tuning `C` parameter between [1.0, 10.0, 100.0]. The best `C` parameter is 1.0.

![alt text][image3]
![alt text][image4]
---

### Video Implementation

#### 1. Link to the final video output
Here's a [link to my video result](./project_video_vehicle_detecting.mp4)
(path: ./project_video_vehicle_detecting.mp4)

#### 2. The filter for false positives and methods for combining overlapping bounding boxes

I recorded the positions of positive detections in recent 5 frame of the video, with `collections.deque`. From the positive detections of the recent 5 iterations I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are five frames and their corresponding heatmaps:
![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all five frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Brief discussion of problems faced in the implementation of this project

1. The threshold for filtering out false positives: it's a simple method to use threshold to filter out the false positives, but sometimes it cannot work properly. If the threshold is set too low, many false positives will be shown on the video, but if the value is too high, no detection will happen. I would design a dynamically changed threshold algorithm for the filter, to make the vehicle detection system more robust.

2. The robustness of the classifier: even though the test accuracy could achieve up to more than 0.99, there would be frames on which vehicles could not be identified. It could be the reason that not enough training data from the video like our project ones. I would put more training data clipped from several frames of the video and train the classifier again to see whether or not this way could improve the performance.

3. Another way to improve the robustness is to replace the classifier with other ones like non-linear SVM or deep neural network. I would try to implement them to compare their performance. 

