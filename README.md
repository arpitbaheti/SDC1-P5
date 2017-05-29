**Vehicle Detection Project**

The goals / steps of this project are the following:

##Feature extraction
####Perform a Histogram of Oriented Gradients (HOG) feature extraction on images
####apply a color transform
####append binned color features
####append histograms of color
####normalize the features

##Training the classifier
####Download dataset
####split dataset into test and train using sklearn train_test_split function
####Extract feature for data (image in our case)
####Train a classifier Linear SVM classifier using sklearn python module

##Run the classifier on test image to detect cars
####Implemented technique described in lessons, which only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows
####Using our trained classifier to search for vehicles in images.
####Use heat map technique for remove noise and multiple window detections.
####Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./my_image/car_nocar.png
[image2]: ./my_image/car.png
[image3]: ./my_image/car_hog.png
[image4]: ./my_image/notcar.png
[image5]: ./my_image/notcar_hog.png
[image6]: ./my_image/test_image_y_start_stop_none.png
[image7]: ./my_image/after_y_start_stop400x660.png
[image8]: ./my_image/scaled_detected.png
[image9]: ./my_image/heat_map.png
[image10]: ./my_image/label_heat_map.png
[video1]: ./test.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Histogram of Oriented Gradients (HOG)

####1. The code for this step is contained in ["lesson_functions.py"](./lesson_functions.py) and defined under get_hog_features() functions usinf sklearn hog function.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

I tried various combinations of parameters and "YCrCB" color space was chosen as HOG is known to perform well in this color space. Other parameter namely orient, pix_per_cell, cell_per_block and hog_channel were chosen heuristically.

Parameters selected were:

```
  color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 9  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  cell_per_block = 2 # HOG cells per block
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```

####2. I trained a classifier using your selected HOG features and color features.

I trained a linear SVM using sklearn LinearSVC() function and used train_test_split() function for splitting the data. (I have kept 20% as testing data) and achieved 99.8% accuracy on test data using the feature extracted. Code for training is present in ["main_code.py"](./main_code.py) file at line number 94 and define under train() function.


###Sliding Window Search

####1. I have implemented a sliding window search.  heuristically I have decide following parameters for sliding window:

```
    y_start_stop = [400, 656]
    overlap = 50%
    xy_window = (96, 96)
```

Following is the image showing detection of Vehicle using Full image sliding window search.

![alt text][image6]

After selecting y_start_stop = [400, 656] the detection improves:

![alt text][image7]

###Scaled image with hog calculation once

####1. I have also used a method described in lessions for for extracting hog features once and then can be sub-sampled to get all of its overlaying windows. This gives better result than silding window as well perform faster.

Follwing is the result of above method:
![alt text][image8]

####2. Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  example image:

![alt text][image10]
---

### Video Implementation

####1. Video pipeline is the same as image pipeline except its run over the entire set of images in the video. Output of the execution could be found at ["test.mp4"](./test.mp4)


####2. I have recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.


### Here are six frames and their corresponding heatmaps and output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and resulting bounding boxes are drawn:
![alt text][image9]

---

###Discussion

####Problem
HOG features are very sensitive to color space and it took time to read up and figure out that YCrCb color space would work best. Other problem I faced was with setting parameters for detecting 'vehicles' these parameters are sensitive and needed lots of experiments to finalize.

####Pipeline Failure
1. Pipeline could fail to work in dark/dim lighting conditions as HOG features would not be sharp in such scenarios. Neural network based approach could be considered to overcome this problem
2. The pipeline is very slow and if deployed in a real system could fail to work in real time deadlines. To solve this problem most of the implementation needs to be optimized and moved to a GPU. HOG feature computation kernel is available on GPU.
3. Another scenario where it could fail is when the vehicles are very far away. The window needs to be scaled to even lower sizes for detecting smaller vehicles. smaller windows however carry a trade-off of increased execution time
4. The pipeline could fail if the vehicle is too close to the camera. This would need training with data containing closer view of vehicles

