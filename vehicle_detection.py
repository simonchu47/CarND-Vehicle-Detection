#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:23:13 2017

@author: simon
"""

import os, fnmatch
import pickle
import cv2
import numpy as np
#import glob
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from useful_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17 use:
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from collections import deque
from sklearn import svm, grid_search, datasets

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_string('training_path', '', "The directory containing training images")
flags.DEFINE_bool('training', False, "Training the classifier or not")
flags.DEFINE_bool('image', False, "Input is image")
flags.DEFINE_bool('video', False, "Input is video")
flags.DEFINE_string('image_path', '', "The input image file path")
flags.DEFINE_string('video_path', '', "The input video file path")
flags.DEFINE_bool('debug', False, "Generate perspective transform matrix")
flags.DEFINE_float('start', 0.0, "The start of the video")
flags.DEFINE_float('end', 0.0, "The end of the video")

# Keeps records of data of left or right lane line mark for each frame
class Vehicle():
    def __init__(self):
        # the number of last iterations to keep
        self.n = 5
        # box list of the last n detections
        self.recent_box_list = deque()
        # sum all the box list
        self.sum_recent_boxes = []
        # heat map of the last n detections
        self.recent_heat_map = deque()
    
    # Keeps the box list for last n iterations
    def keep_last_iterations(self, box_list):
        if box_list == []:
            if len(self.recent_box_list)>0:
                self.recent_box_list.pop()
        else:
            if len(self.recent_box_list)<self.n :
                self.recent_box_list.appendleft(box_list)
            else:
                self.recent_box_list.pop()
                self.recent_box_list.appendleft(box_list)
            
    # Conconate all the rescent box list
    def sum_all_boxes(self):
        if len(self.recent_box_list)>0:
            self.sum_recent_boxes = np.concatenate(self.recent_box_list)
    
    # Keeps the heat map for last n iterations
    def keep_heat_map(self, heat_map):
        if len(self.recent_heat_map)<self.n:
            self.recent_heat_map.appendleft(heat_map)
        else:
            self.recent_heat_map.pop()
            self.recent_heat_map.appendleft(heat_map)
vehicle_detect = Vehicle()        

### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
#y_start_stop = [360, None] # Min and max in y to search in slide_window()
    
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# The process of training a classifier
def train_classifier(data_path):
    #images = glob.glob(data_path+'/'+'*.png')
    images = find('*.png', data_path)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image in image:
            notcars.append(image)
        else:
            cars.append(image)
    
    # Extract features for cars
    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    # Extract features for NOT cars
    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    if FLAGS.debug == True:
        # Randomly choose a car picture for test and showing results
        car_test_img_file = random.choice(cars)
        if car_test_img_file.split('.')[-1] == "png":
            readin = cv2.imread(car_test_img_file)
            image = cv2.cvtColor(readin, cv2.COLOR_BGR2RGB)
        else:
            image = mpimg.imread(car_test_img_file)
        
        file_name = car_test_img_file.split('/')[-1].split('.')[0]
        
        single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,
                            debug=True, name=file_name)
        
        # Randomly choose a non-car picture for test and showing results
        notcar_test_img_file = random.choice(notcars)
        if notcar_test_img_file.split('.')[-1] == "png":
            readin = cv2.imread(notcar_test_img_file)
            image = cv2.cvtColor(readin, cv2.COLOR_BGR2RGB)
        else:
            image = mpimg.imread(notcar_test_img_file)
        
        file_name = notcar_test_img_file.split('/')[-1].split('.')[0]
        
        single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,
                            debug=True, name=file_name)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
        
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # The following is parameter setting for optimizer
    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} 
    #parameters = {'kernel':['linear'], 'C':[1, 10, 100]} 
    #svr = svm.SVC()    
    # Use a linear SVC 
    svc = LinearSVC()
    #svc = grid_search.GridSearchCV(svr, parameters)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    # Show the best parameter of the optimization
    #print(svc.best_params_)
    return svc, X_scaler

# The pipeline to process each picture or frame
def pipeline(image):
    
    filename = FLAGS.image_path.split('/')[-1].split('.')[0]
    pos_box_list = []
    
    # The following is for large scanning window searching
    ystart = y_start_stop[0]
    ystop = y_start_stop[1]
    scale = 1.8
    
    if FLAGS.debug == True:
        positive_boxes, out_image = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              color_space, spatial_feat, hist_feat, hog_feat, hog_channel, debug=True)
        cv2.imwrite('./'+filename+'_large_window.jpg', cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    else:
        positive_boxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              color_space, spatial_feat, hist_feat, hog_feat, hog_channel, debug=False)
   
    if len(positive_boxes)>0:
        pos_box_list.append(positive_boxes)

    # The following is for middle scanning window searching
    ystart = 400
    ystop = 500
    scale = 1.0
    
    if FLAGS.debug == True:
        positive_boxes, out_image = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              color_space, spatial_feat, hist_feat, hog_feat, hog_channel, debug=True)
        cv2.imwrite('./'+filename+'_mid_window.jpg', cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    else:
        positive_boxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              color_space, spatial_feat, hist_feat, hog_feat, hog_channel, debug=False)
    
    if len(positive_boxes)>0:
        pos_box_list.append(positive_boxes)
    
    if len(pos_box_list) > 0:
        pos_box_list = np.concatenate(pos_box_list)
    
    # Keep record the box list for this iteration
    vehicle_detect.keep_last_iterations(pos_box_list)
    # Accumulate all the box lists for the last 5 iterations
    vehicle_detect.sum_all_boxes()
    recent_boxes = vehicle_detect.sum_recent_boxes
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    if len(recent_boxes)>0:
        # Add heat to each box in box list
        heat = add_heat(heat, recent_boxes)
        
        # Apply threshold to help remove false positives
        if FLAGS.image is True:
            heat = apply_threshold(heat, 0)
        if FLAGS.video is True:
            heat = apply_threshold(heat, 3)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
    
        #if FLAGS.debug == True:
        #    output_image = np.dstack((heatmap, heatmap, heatmap)).astype(np.uint8)
        #    cv2.imwrite('./'+filename+'_heatmap.jpg', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        
        output_image = draw_labeled_bboxes(image, labels)
        
        if FLAGS.debug == True:
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(output_image)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            if FLAGS.image is True:
                plt.savefig('./'+filename+'_heatmap.jpg', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
                plt.close()
            if FLAGS.video is True:
                vehicle_detect.keep_heat_map((output_image, heatmap))
                plt.close()
                t = time.time()
                # Show all the car positions and heat map for the last 5 iterations
                fig2 = plt.figure(figsize=(6.4,14.8))
                for n in range(len(vehicle_detect.recent_heat_map)):
                    plt.subplot(5,2,n*2+1)
                    plt.imshow(vehicle_detect.recent_heat_map[-n-1][0])
                    plt.title('Car Positions')
                    plt.subplot(5,2,n*2+2)
                    plt.imshow(vehicle_detect.recent_heat_map[-n-1][1], cmap='hot')
                    plt.title('Heat Map')
                
                fig2.tight_layout()
                plt.savefig('./'+filename+str(t)+'_heatmap.jpg', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
                plt.close()
    else:
        output_image = image
    
    if FLAGS.debug == True:
        if FLAGS.video is True:
            t = time.time()
            fig = plt.figure()
            plt.imshow(labels[0], cmap='hot')
            plt.savefig('./'+filename+str(t)+'_labels.jpg', transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
            plt.close()
            cv2.imwrite('./'+filename+str(t)+'_output_bbox.jpg', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        
    return output_image

if FLAGS.training is True:
    if FLAGS.training_path is not "":
        print("Doing the training...")
        svc, X_scaler = train_classifier(FLAGS.training_path)
        svc_pickle = {}
        svc_pickle["svc"] = svc
        svc_pickle["orient"] = orient
        svc_pickle["pix_per_cell"] = pix_per_cell
        svc_pickle["cell_per_block"] = cell_per_block
        svc_pickle["spatial_size"] = spatial_size
        svc_pickle["hist_bins"] = hist_bins
        svc_pickle["color_space"] = color_space
        svc_pickle["hog_channel"] = hog_channel          
        svc_pickle["spatial_feat"] = spatial_feat
        svc_pickle["hist_feat"] = hist_feat
        svc_pickle["hog_feat"] = hog_feat          
        svc_pickle["X_scaler"] = X_scaler
        # Save the classifier and the related parameters into a pickle
        pickle.dump( svc_pickle, open( "./trained_svc_pickle.p", "wb" ) )
    else:
        print("Please add the training_path flag...")
else:
    # Read in the trained classifier and the related parameters
    trained_svc_pickle = "./trained_svc_pickle.p"
    if os.path.isfile(trained_svc_pickle) is True:
        print("Find the pickle.........")
        svc_pickle = pickle.load(open(trained_svc_pickle, "rb"))
        svc = svc_pickle["svc"]
        orient = svc_pickle["orient"]
        pix_per_cell = svc_pickle["pix_per_cell"]
        cell_per_block = svc_pickle["cell_per_block"]
        spatial_size = svc_pickle["spatial_size"]
        hist_bins = svc_pickle["hist_bins"]
        color_space = svc_pickle["color_space"]
        hog_channel = svc_pickle["hog_channel"]
        spatial_feat = svc_pickle["spatial_feat"]
        hist_feat = svc_pickle["hist_feat"]
        hog_feat = svc_pickle["hog_feat"]
        X_scaler = svc_pickle["X_scaler"]
    else:
        print("Sorry!, no trained svc pickle...")
        
def main(__):
    # Do the classifier training first,
    # then do the vehicle detecting for one picture or video   
    if FLAGS.training == False:        
        if FLAGS.image is True:
            if FLAGS.image_path is not "":
                if os.path.isfile(FLAGS.image_path) is True:
                    filename = FLAGS.image_path.split('/')[-1].split('.')[0]
                    img = mpimg.imread(FLAGS.image_path)
                    result = pipeline(img)
                    # Output the result
                    cv2.imwrite('./'+filename+'_vehicle_detecting.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                else:
                    print("There is no input image...")
            else:
                print("Please add the input image path...")
        elif FLAGS.video is True:
            if FLAGS.video_path is not "":
                if os.path.isfile(FLAGS.video_path) is True:
                    filename = FLAGS.video_path.split('/')[-1].split('.')[0]
                    if FLAGS.start == FLAGS.end:
                        clip1 = VideoFileClip(FLAGS.video_path)
                    elif FLAGS.start < FLAGS.end:
                        clip1 = VideoFileClip(FLAGS.video_path).subclip(FLAGS.start,FLAGS.end)
                    else:
                        print("Start time is less than end time!!!")
                    drawing_lane_clip = clip1.fl_image(pipeline)
                    drawing_lane_clip.write_videofile('./'+filename+'_vehicle_detecting.mp4', audio=False)
                else:
                    print("There is no input video...")
            else:
                print("Please add the input video path...")
        else:
            print("Please add image or video flag...")
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
