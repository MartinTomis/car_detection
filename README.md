# Vehicle detection

In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. 

# README

**1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.**

Below....

# Histogram of Oriented Gradients (HOG)
**1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.**

I leveraged the functions presented in lectures. 
Feature extraction is performed on lines 165 and 166 for car and non-car images by function extract_features. This function calls function get_hog_features, which extracts features for the selected color channel (lines 148-150). The implementation uses all 3 channels. The parameters for adjust are the following: 
* pix_per_cell - this is the number of pixels per "cell" - a unit for which the gradient is calculated. It is passed as a tuple (e.g. *(8,8) means 8 x 8 pixels).
* cell_per_block - number of cells per block. The histogram calculated for each cell may be normalized, and the cell_per_block determines the set of cells, over which the normalization is computed. 
* orientations - this means the number of "bins", into which the gradient angles are allocated. The

The training images are all 64 x 64, so separating each picture into  8 x 8 cells, each 8 x 8 pixels large, seems straightforward, and it is the same setting as done in the lecture. For cell_per_block, I go for (2,2), meaning that the normalization is performed over the area of 2 x 2 adjacent cells (and then this block is shifted by 1 cell).

I set the number of orientations into 9, as done in lecture. This means that the possible 360 degree range is split into 9 bins by 40 degrees - this sounds as a reasonable level of granularity.


**1. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**
I used both HOG feature and color features, leveraging code from lectures. The features are extracted for both car and non-car images in lines 165 and 166, and are normalized in rows 177-182. Normalization improves numerical optimalization used by the machine learning algorithm and controls to some degree for outliers in the data.

I use support vector machine with a radial basis kernel. This is a linear-classifier with a "kernel trick": it allows to create a non-linear boundary to the original data, by generating additional features (simple functions of the original features) which are linearly separable. Besides the kernel specification, I used the default values for parameters "C" and "gamma", which control for the non-linearity of the boundary (after the kernel trick) and the strength of effect of observations far from the boundary, respectively. I attepted to optimize the parameters by using GridSearchCV, but the results did not seem superior to the choice of the default values, in particular on the images from the video.

The training is performed on lines 199-205. Assessment of the algorithm by analyzing the accuracy (99+%) on a testing set is performed on line 209.


# Sliding Window Search

**1.Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?**
sdsdds

**2. Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier? **
The classifier performance is very good on the test data sampled from the data used for training. 

However, my biggest concern was how the classifier generalized on the images from the video. Moreover, the "car detector" should identify the car if only a part of the car appears, rather than the whole car. 

In an effort to improve this, I tried to train the classifier on more images, sourced from cropping and then resizing the training images. The idea is that this would ideally allow the classifier to recognize as a "car" only e.g. the first half of a car. I used the following code (mostly commented out in the final code - lines 43-49 and 57-63) following:

```python
        img=mpimg.imread("non-vehicles/" + dir + "/" + image_name)
        img1=cv2.resize(img[0:40, 0:40,:], (64,64))#img[0:40, 0:40].resize() #cv2.resize(img[:, :, 0], size)
        img2 = cv2.resize(img[23:63, 23:63,:], (64, 64))
        non_car_list.extend([img, img1, img2])
```

It did improve something... For example, the code was able to recognize the car before it all appeared in the frame:
![alt tag](https://github.com/MartinTomis/car_detection/blob/master/car_part.png)

However, there were some downsides to this as well. The prediction of the images was very slow (I found this quite surprising, as the number of feature extracted from each image before training was the same). More importantly, lead to a high number of "true-negatives", i.e. case where a car was not recognized. I however think that this could be remediated by more "granular" sliding window. However, given the time it took to predict, more granual sliding was not feasible option for me, without substantially changing the code (parallelization would probablu have to be necessary).

I hence used the input images without cropping.





# Video Implementation

**1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.) **
dsdsd

**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes. **

# Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?**

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
