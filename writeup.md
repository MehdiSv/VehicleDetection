**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car.png
[image2]: ./writeup_images/non-car.png
[image3]: ./writeup_images/car_hog.png
[image4]: ./writeup_images/non-car_hog.png
[image5]: ./writeup_images/windows.png
[image6]: ./writeup_images/hot_windows.png
[image7]: ./writeup_images/heatmap.png
[image8]: ./writeup_images/labels.png
[image9]: ./writeup_images/final_boxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### HOG

#### 1. HOG features extraction for learning

I used all the images from the vehicles and non-vehicles folder (lines 109 to 114 of `main.py`).

I tried several combinations of HOG parameters, in particular the color space and channel, for learning (lines 10 to 14 of `solution.py`).

Here are some base images and their HOG results using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car: ![Car][image1] => ![Car HOG][image3]

Non-car: ![Non-car][image2] => ![Non-car HOG][image4]

#### 2. Final choice of HOG parameters.

Changing the number of orientations, of pixels per cell and cells per block never gave significantly better results.
So I mainly tweaked the color space and channel used, in particular RGB (red or all channels), HSV (every combination) and YCrCb (all channels)
In the end YCrCb on all channels gave the best results, with the original HOG settings: 8 orientations, 8\*8 pixels per cell and 2\*2 cells per block

My settings for HOG can be found in the lines 10 to 19 of `solution.py`.

The methods computing the hog are `compute_hog_features()` in `solution.py` and `get_hog_features()` in `lesson_functions.py`

#### 3. Classifier training

I did the training using a pretty standard method:
- I first standardized the computed features using a standard scaler
- Then I split my data into training and test sets (80/20), randomizing them in the process
- I finally trained a standard SVC with default settings

I also stored the results of the computed features, scaler and SVC to pickle files to avoid having to recompute them every time.

All of this can be found in the `compute_features`, `learn_scale` and `learn` functions of `main.py`

### Sliding Window Search

#### 1. Sliding windows

My first attempt was with only 64\*64 windows all over the image. This obviously didn't give great results as I was detecting cars even in the sky.

I then restricted the windows to the lower half of the image. I also added different window sizes, ranging from 64\*64 to 128\*128.

Based on the size of the window, I also further restricted their location. For example, 64\*64 windows are only computed on the top half of the lower half of the image
as cars located on the lowest part of the image are supposed to be up close and thus appear bigger.

Finally, I made my window overlap a lot. This allows me to set a higher threshold on my heatmap which gave me significantly better results, detecting more true positives and less false positives.

Sliding windows setup can be found in lines 142 to 159 of `main.py` and in the `slide_window()` function in `lesson_functions.py`

![Windows][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To summarize, I searched on three window sizes using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.
I optimized the performance of the classifier by computing the HOG features once on the whole image and then extracting the region of interest for each window.

The pipeline can be found in the `process_image()` function of `main.py`.

An example of the resulting detection:

![alt text][image6]
-------------------

### Video Implementation

#### 1. Final video output.
Here's a [link to my video result](./project_video_done.mp4)


#### 2. False positives filter

I tried to filter false positives by saving the detection results over the last five frames using a circular buffer and applying a threshold on the resulting heatmap.
I then used scipy's `label()` function to label each car and draw a bounding box around it.

The heatmap computing can be found in the `Heatmapper` class and the labelling in the function `process_image()` of `main.py`

### Here is an example heatmap:

![Heatmap][image7]

### Here is the labelled heatmap:

![Labels][image8]

### Here the resulting bounding boxes:

![Final bounding boxes][image9]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is likely to fail in worse weather/lighting conditions. It could also fail should there be less contrast between the cars and the road in general.

The pipeline will lose track of a car when another car passes in front it as I don't store any historical bounding box or car information to try to do some prediction.
I think this would be a great improvement to the implementation as it would avoid jittery bounding boxes, prevent false positives and help with punctually bad conditions (image
quality, car hidden by another one, etc).
