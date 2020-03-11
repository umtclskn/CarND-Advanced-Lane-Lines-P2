
##  Self-Driving Car Engineer Nanodegree Program






***Advanced Lane Finding Project***

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboard_image]: ./output_images/undistort_output.png
[undistorted_image]: ./output_images/undistorted_chessboard.png 
[calibrated_image]: ./output_images/undistorted_sample.png
[sobel_x]:  ./output_images/sobel_x.png
[mag_binary]:  ./output_images/mag_binary.png
[dir_binary]:  ./output_images/dir_binary.png
[hls_binary]:  ./output_images/hls_binary.png
[combined_binary]:  ./output_images/combined_binary.png
[warped_img]:  ./output_images/warped_img.png
[sliding_window]:  ./output_images/sliding_window.png
[histogram_image]:  ./output_images/undistorted_sample.png
[curved_image]:  ./output_images/curved_image.png
[video1]: ./project_video.mp4 "Video"


---
### Camera Calibration

#### 1. How do we compute the camera matrix and distortion coefficients?

Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are.

OpenCV functions: [findChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners) , [drawChessboardCorners()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners) and [calibrateCamera](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera)
 to automatically find and draw corners in an image of a chessboard pattern. 
 * Store the camera matrix (mtx) & distortion matrix in 'camera_cal/mtx_dist_pickle.p' file
![alt text][chessboard_image]
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][undistorted_image]


### Pipeline

#### 1. Example of a Distortion-Corrected Image.

There are two main steps to this process: use chessboard images to obtain image points and object points, and then use the OpenCV functions `cv2.calibrateCamera()` and `cv2.undistort()` to compute the calibration and undistortion.

I apply this functionality to all images under the 'test_images' folder.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][calibrated_image]

#### 2. HlS Color Space, Gradient, Magnitude of Gradient, Direction of Gradient Threshold & Combine  Them

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in Advanced_Lane_Finding.ipynb).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][hls_binary]
![alt text][sobel_x]
![alt text][mag_binary]
![alt text][dir_binary]
![alt text][combined_binary]


#### 3. Performed a Perspective Transform and Provide an Example of a Transformed Image.

The code for my perspective transform includes a function called `warp()`, which appears in the Advanced_Lane_Finding.ipynb. The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([(570,470),
                  (750,470),
                  (250,685),
                  (1125,685)
                  ])

    dst = np.float32([(250,0),
                  (950,0),
                  (250, 720),
                  (950,720)
                  ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570,470      | 250, 0        | 
| 750,470      | 950, 0|
| 250,685    | 250, 720      |
| 1125,685      | 950, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_img]

#### 4. Identified Lane-line Pixels and Fit Their Positions With a Polynomial?

As shown in the previous animation, we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go.

> histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

* Split the histogram into two sides, one for each lane line.
* Find out which activated pixels from `nonzeroy` and `nonzerox` above actually fall into the window.
* Next step is to set a few hyperparameters related to our sliding windows, and set them up to iterate across the binary activations in the image.
* Iterate through  `nwindows`  to track curvature

### Note: We are using `find_lane_pixels()` in step by step detection, but we are going to change or evaluate this function as `sliding_windows_find_lanes_coeffs()` in Line class.


![alt text][sliding_window]

#### 5. Calculation the radius of curvature of the lane and the position of the vehicle with respect to center.
Our camera image has 720 relevant pixels in the y-dimension.

I'll say roughly 700 relevant pixels in the x-dimension (our example of fake generated data above used from 200 pixels on the left to 900 on the right, or 700).

to convert from pixels to real-world meter measurements, we can use:
>ym_per_pix = 30/720 # meters per pixel in y dimension
>xm_per_pix = 3.7/700 # meters per pixel in x dimension

Located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve:
> f(y)=A*y^2+By+C

[ Radius of Curvature details]([https://www.intmath.com/applications-differentiation/8-radius-curvature.php](https://www.intmath.com/applications-differentiation/8-radius-curvature.php))

#### 6. Result plotted back down onto the road such that the lane area is identified clearly. 
All We have to do is unwrap the image and add the text on it.



![alt text][curved_image]

---

### Pipeline (video)

#### It's useful to define a `Line()` class in Advanced_Lane_Finding.ipynb file to keep track of all the interesting parameters you measure from frame to frame. 
> class Line():

And using the sliding_windows_find_lanes_coeffs functions for tracking and managing the line detection.

This Function is processing and detecting lines in image
> def pipeline (img):

Here's a [link to my video result](./[project_video_output.mp4])

---

### Discussion

#### 1. Where will your pipeline likely fail?  What could you do to make it more robust?
Pipeline works on the standart road. It fails on heavly curved roads or it fails if a car in front of our car.
It fails on different resolutions videos or images.
The camera sight is important for assign src point for ROI and wrap points but I set this field staticaly if camera sight change, lane will not detectable.

Potential improvement:
* Dynamically detect the src point.
* Dynamically detect the threshold parameters.
* Implement what happened is a car in front of the our car
* Implement what happened if car change line

Deep learning techniques may be useful for solving these situations.