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


**2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**
I used both HOG feature and color features, leveraging code from lectures. The features are extracted for both car and non-car images in lines 165 and 166, and are normalized in rows 177-182. Normalization improves numerical optimalization used by the machine learning algorithm and controls to some degree for outliers in the data.

I use support vector machine with a radial basis kernel. This is a linear-classifier with a "kernel trick": it allows to create a non-linear boundary to the original data, by generating additional features (simple functions of the original features) which are linearly separable. Besides the kernel specification, I used the default values for parameters "C" and "gamma", which control for the non-linearity of the boundary (after the kernel trick) and the strength of effect of observations far from the boundary, respectively. I attepted to optimize the parameters by using GridSearchCV, but the results did not seem superior to the choice of the default values, in particular on the images from the video.

The training is performed on lines 199-205. Assessment of the algorithm by analyzing the accuracy (99+%) on a testing set is performed on line 209.


# Sliding Window Search

**1.Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?**
I implemented the sliding window search iby function "find_cars". I modified the function from lecture slightly, so that the output is a list of the box vertices, rather than the image. 



I also defined additional inputs: xstart and xstop coordinates, number of cells per step and a list of vertices. The xstart and xstop are parameters that helped with the implementation in this given video, where the car drives on a highway and all relevant cars are on the right side of the frame, and it was very inefficient to search the left hand side, apart from a small strip for incoming traffic.


When looking for "medium" and "small" cars, i.e. cars that are either far away or in a medium distance, I slided by 1 cell. When looking for larger cars, by 2.

The last new input is the list of the box vertices, to which new vertices are appendend, if cars are found. I did this, so that multiple dimensions of the sliding windows may be used. 

The choice for the sliding window is shown below, with comments

    #small cars ahead - very granular search, for cars fitting roughly into "64 x 64 box"
    boxes_to_draw = find_cars(image, 396, 492, 720, 1000, 1, svc1, X_scaler1, 9, 8, 2, 1, (32, 32), 32, boxes_to_draw)

    # search for mid-sized cars: The algorithm looks for cars images of roughly 96 x 96 dimensions (closer to center) or cca 128 x 128 -    farther from center
    boxes_to_draw = find_cars(image, 400, 528, 800, 1104, 1.5, svc1, X_scaler1, 9, 8, 2, 1, (32, 32), 32, boxes_to_draw)
    boxes_to_draw = find_cars(image, 400, 560, 976, 1280, 2, svc1, X_scaler1, 9, 8, 2, 1, (32, 32), 32, boxes_to_draw)

    #large near cars
    boxes_to_draw = find_cars(image, 400, 656, 1024, 1280, 3, svc1, X_scaler1, 9, 8, 2, 2, (32, 32), 32, boxes_to_draw)
    
  
  Below is 
  ![alt tag](https://github.com/MartinTomis/car_detection/blob/master/test_1.png)



**2. Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?**
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

The code hence does not use the extra images created by cropping and resizing, though I think the idea is not bad.

# Video Implementation

**1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)**

Here's a [link to my video result](https://github.com/MartinTomis/Lane_detection/blob/master/video.mp4)

**2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

I implemented a filter for false positives based on the example in lectures. Function "add_heat" takes as input the list of boxes and adds value of 1 for each pixel within the boxes. Where the overal value is below or equal to certain threshold, the values is set to 0.

Besides false positives, a problem are true negatives, i.e. cases where what should be a car is not identified as one.

I dealt with these 2 problems in the folowing manner:
* The threshold in add_heat is set to 1. This essentially means that I draw a box if classify the image as a car, i.e. the value of 1 does not really prevent false positives. Given how I restricted the sliding window only to the region where the cars are in the video, this was sufficient, but I would use higher threshold for a different setting.
* To avoid true negatives, I combine the current image with 3 previous images (where available) to create heatmap. I was considering multiple options. Ideally, I would want to assign a lower weight to more distant images. However, the values we are potentially dealing with may be low integers, quite potentially 0s and 1s. I hence simply add the heatmatps for the 3 previous images, withoug any more advanced weighting. 

    heatmap=add_heat(result.shape, boxes_to_draw, 1)
    if len(box_list)>3:
        heatmap1 = add_heat(result.shape, box_list[-2], 1)
        heatmap2 = add_heat(result.shape, box_list[-3], 1)
        heatmap3 = add_heat(result.shape, box_list[-4], 1)
   
 The disadvantage of this approach, is that if the position of the car in from of the camera is changing, then 

I combine overlapping bounding boxes by using the label and draw_labeled_bboxes functions. Both were shown in the lectures.
The label function takes the output of the add_heat function, and identifies multiple clusters in the data, and draw_labeled_bboxes



# Discussion

**1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?**

* Speed of prediction. It is by far the biggest problem. I think that unless the cars can be detected in less than 1/10 sec, the "car detector" is not working properly. I tried "multiprocessing" python library, but could not get it running, and did not attempt parallelization further. Once speeded up, more granular search over larger area could be implemented, highlighting incoming cars or using classifier trained even on the cropped training images.
* I had more problems with true negatives than false positives. To avoid true negatives, it makes sense to use information from previous images. However, my implementation leads to certain "lag", drawing the box also over images where the car was recently. This could be done by adjusting the information from the previous images - ideally by shifting them by few pixels in the direction inferred from the difference between our speed and the other car speed.
