##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/single_frame.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/heatmap_thresholded.png
[image6]: ./output_images/draw_image.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The Jupyter notebook contains the code for this project.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function `get_hog_features()` extracts HOG features from an image. This function is called from the wrapper functions `extract_features()`, used for training on a number of images, and `single_img_features()`, used in the detection pipeline to extract features from a single frame of the video.

I started by reading in all the `vehicle` and `non-vehicle` images using `glob`.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I tried a few HOG parameters and reviewed the notes and Q+A from Udacity.
- The color space YCrCb was recommended in the Q+A, and I noticed significant improvement over RGB
- 9 orientation bins is mentioned as a number that yields good performance in the HOG literature
- 8 pixels per cell and 2 cells per block I kept constant from the class, and found them to be effective
- I also tried with individual channels and all channels for HOG, and all channels gave me some increased accuracy.

That said, the accuracy did not change by a great deal over the course of my experimentation. Ultimately, the proof was in the superior performance on the video once I switched to YCrCb and used all channels for my HOG features.

Here is an example of the the output of extracting HOG parameters on the Y channel for sample images:

![alt text][image2]

I also checked for accuracy with and without the spatial and color histogram features, and accuracy was superior with these both on. For a production use case, I think I would automate trying different parameters and determining which yield the highest accuracy.

Final parameters used for training my classifier are in the "Training a classifier" section.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the "Training a classifier" section, I trained a linear support vector classifier (SVC). The process is similar to the class, where I:
- Extract features from images of vehicles and non-vehicles
- Split into training/test
- Scale the data (as the different types of features may be at different scales)
- Fit the SVC on the training data
- Check the accuracy on the test data

The only difference from the class is that to fit the scaler, I only used training data. I'm not sure if it would have made a noticeable impact to use test data too, but I wanted to split off the test data before learning anything at all about it. I recalled from some video either with Andrei Karpathy or Andrew Ng where they mention it's a common mistake to include test data in steps such as determining normalization parameters, and that leaks test data into the training, causing overfitting.

Having fit the scaler on the training data, I then of course applied it to both training and test data (and features that I extract in my pipeline).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function `slide_window()` returns a list of windows based on an image, a region of the image as defined by x_start_stop and y_start_stop, window size, and window overlap. I mostly implemented this like the lectures, except that I believe the number of windows calculation is not correct, as I put in this post: https://carnd-forums.udacity.com/questions/36901421/sliding-window-implementation-%23-of-windows

The sizes of windows I chose were between the smallest size I would effectively need, up to the largest size a car appears as it pulls up next to you. Those turned out to be: (72,72),(96,96),(128,128),(160,160),(196,196). I also chose an overlap of 75%. This was calibrated by a lot of manual testing, trading off how long it takes to process vs how good the results were, and 75% gave a reasonable balance.

Note that in my pipeline, for each window size, I only searched for one "row" of windows. This is because for a given car, as it gets further away, the top of the car stays at a consistent height in the image, although the bottom of the car keeps changing. This gave me some savings from a performance standpoint, because it reduces the number of windows. I rely on using more window sizes to capture closer cars. It worked pretty well for me, thought I would want to spend some time in the future considering the potential cons of using this approach (for example, tall cars). This is captured in the code Pipeline.process_frame().

I spent the most time on this project thinking of different approaches to windowing, as well as handling overlaps and false positives, trying this out on small test videos that were problematic. In a section below I talk more about this.

The function `search_windows()` takes the list of windows, extracts features, and makes a prediction based on the trained classifier. It returns the list of windows classified as containing a car.

Here is an example of windows I would look at:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the windows (72,72),(96,96),(128,128),(160,160),(196,196), using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Performance was sufficient for me, although in the future I might try the approach of extracting features once per image, and potentially downsampling the image.

Here is an example of the pipeline, with a low threshold (see below) such that some bounding boxes are visible:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I went through a variety of pipelines that help with false positives and overlaps before settling in on my final one. 

Ultimately, my pipeline is in the class `Pipeline()`.
- Instantiating this class takes the parameters `xy_windows`, `xy_overlap`, `heat_frames`, and `heat_threshold`.
- heat_frames is how many frames of the video I consider at a time. Through experimentation I arrived at 7.
- heat_threshold is the required threshold of heat for an area to be considered part of a car. Through experimentation I arrived at 6.
- recent_heatmaps is the time series of heatmaps to be considered
- heatmap is the placeholder for an image-sized heatmap of zeroes

The pipeline is in the method `process_frame()`, which is called for every frame of the video:
- Once the image size is known, I set heatmap to zeroes and fill up recent_heatmaps with these heatmaps, up to the number of frames to consider (heat_frames)
- Loop through each window size to be used
- Get the windows with a predicted car using `slide_window()` and `search_windows()` as described perviously
- Put all of these windows in a list of 'hot' windows.

Now, the approach I used to turn these windows into bounding boxes is as follows:
- First, I create a frame heatmap of zeros and use `add_heat()` to add 1 to the heatmap at the locations of all the hot windows.
- Next, I turn that heatmap into a binary heatmap, where any pixel with at least a value of 1 is set to exactly 1. I discuss this further below.
- I update my list `recent_heatmaps` with this new binary heatmap.
- I apply the heat_threshold using `apply_threshold()` to keep only those areas that are hot enough.
- As in the class, I get the labels from this thresholded heatmap to get the blobs where there might be cars, and draw boxes.

Binary heatmap approach: Initially, I used non-binary heatmaps, which give greater weight to areas of overlap within a frame. I tried various thresholds, but ultimately I found that either I chose a threshold that was too low (and gave false positives) or too high (didn't capture enough of the car, just the center of the car with lots of overlaps). I thought it might be better to just focus on those areas which consistently appear to have cars over several frames, and not weigh too highly overlaps within a frame. In this way, there's effectively a 2-threshold approach. My threshold within a frame is 1. My threshold across frames is 6. This was pretty effective.

Another approach I tried early, and would want to experiment more with is treating the different window sizes separately. I think I might be able to get more tight bounding boxes if I don't consider overlaps of windows of different sizes.

Here's an example result showing the binary heatmaps for several frames...as they are binary, they are pretty boring. Sorry about that, but it works pretty well!:

![alt text][image5]

The resulting heatmap after thresholding is then shown here:

![alt text][image6]

The resulting of the pipeline is then:

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are serveral improvements I would consider making:
- Reviewing literature for how to handle partial car detections. Perhaps if I broke the car images up into partial-cars, I could better detect cars at the edges of the image which are partially obscured. Currently I don't do well with this.
- Smoother bounding box movement over time. I couldn't think of a way to do this without correlating labels in my heatmap across frames, and it felt like part of a larger project :)
- Get tighter bounding boxes, which I think can potentially come from either looking a partial car areas or by considering each window size separately for overlaps.
- Performance improvements such as extracting features once, so I can iterate faster and so it would perform better in real-life. Once second per frame is obviously not acceptable for running in a car.