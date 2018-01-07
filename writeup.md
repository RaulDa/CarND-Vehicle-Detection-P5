## Project 5. Vehicle detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image2]: ./output_images/hog.png
[image3]: ./output_images/predictions.png
[image4]: ./output_images/filter_false_positives.png

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.   

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The defined function for implementing this step is contained in the section 1, cell 4 (HOG) of the code contained in the IPython notebook (`P5.ipynb`). The extraction from the training images is performed in the section 2, cell 2 (Feature extraction).

I started by reading in all the `vehicle` and `non-vehicle` images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Regarding `pixels_per_cell` I tried the values 4, 8 and 16, since these are submultiples of 64 (the defined size of the sliding window). I took care of the values in combination of `cells_per_block` (I tried to not choose high values for both). With 16 pixels per cell, the performance of the classifier did not improve in comparation with the performance obtained with 8. With 4, my processor even got stucked, so I just decided to use 8.

Regarding `cells_per_block`, I used the values 1, 2 and 3. With all values the performance on the training and test sets was similar, however, with 1 cell per block (no normalization), more false positives were detected on the video, and 3 increased the number of features without clearly improving the final detection. So I decided to take the value 2.

I took 8 `orientations` for the current configuration. Other values resulted in a better performance with other configurations of `pixels_per_cell` and `cells_per_block` (for example, 6 orientations with 16 pixels per cell and 1 cell per block), but with the current configuration 8 resulted in less false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training of the classifier is located in all cells of the section 2 of the IPython notebook (Training of classifier).

I extracted HOG, binned color (32x32 image) and color histogram (32 bins) features, concatenating them in a single array and repeated the process for all training images. I introduced the features in a linear Support Vector Machine for training.

I tried to use non-linear kernels for the SVM, like rbf or sigmoid, expecting better results than a linear one. However, the execution through the function `SVC` took quite longer than using the linear SVM through `LinearSVC`. Taking into account that the accuracy for the linear was 0.99, I finally discarded the use of non-linear ones.

I used a value of the `C` parameter of 2. I expected this parameter to have a decisive influence in the detection of false positives, but after trying with several values, I concluded that in this case the influence was low. Since I tried several powers of two, I finally chose 2 as final value.

Regarding the training data, I used both GTI and KITTI datasets, extracting 14060 images. I did a deterministic selection of images for the test set, by taking 1762 vehicle images of the KITTI dataset and 1762 extra non-vehicle images. I did not randomize the selection of the test data because the GTI data is mostly composed of sequences of almost identical images. Taking random images could make that there are identical images for training and test set, so that the accuracy data would be not trusted. Once the test images are selected, I randomize the order before passing them to the SVM.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is located in the `find_cars` function of the section 3 (Prediction on single image).

I take 64 as window size and take into account the pixels per cell and the cells per block to calculate the number of steps in both X and Y axis, in function of the size of the image.

I decided to search random window positions at scales 1.0, 1.5 and 2.0. I went over the Y positions 350 to 690 for the scales 1.5 and 2.0, and over the positions 350 to 520 for the scale 1.0. Higher values had low influence in the detection.

I finally chose to do the HOG extraction for the entire frame and not for each window, and just obtain the result of the single extraction corresponding to the current window.

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here is an example of the result without applying filtering of false positives:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map (with a value of 5) to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

For the video, I took into account the 25 previous frames to build a mean heat map. I took the mean heat map to calculate the final boxes in each frame.

Here's an example result showing the heatmap from a frame after filtering undesired false positives, and the final bounding boxes represented:

![alt text][image3]

![alt text][image4]




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The attempt of training using a non-linear kernel represented an initial issue. I invested some time configuring the GridSearchCV function and the C and gamma parameters, but the use of the SVC function made that the training take a huge amount of time. Since the linear kernel gives quite good results, I finally chose this configuration. If I had to improve the pipeline, I probably would choose a non-linear kernel (rbf) with the best parameter configuration to get a bit more accuracy.

The read of images was also an issue at the beginning. I finally used openCV for both .png (training) and .jpg (video), and converted from BGR to YCrCb.

The actual detection on the video detects some false positives that are shown and could not be filtered. This probably could be avoided by increasing the number of previous frames to be considered when applying the threshold, but with the consequence that probably less detections of the white car are achieved. In fact, with the current configuration the white car is not detected for a tiny amount of time.
