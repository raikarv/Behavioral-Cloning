# **Behavioral Cloning- P3** 

## Report

### Test data test for Track 1 provided is used to train and simulate driving on the track

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* ReportP3.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

MY convolutional neural network architecture was inspired by NVIDIA's End to End Learning for Self-Driving Cars paper. One of the difference I used only four Convolutional layers with each layer followed by MaxPooling layers just after each Convolutional Layer in order to cut down training time. The model includes RELU layers for nonlinearity, and the data is normalized using a Keras lambda layer.Then this was followed by 5 fully connected layers

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on the given data set to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with forced learning rate 1e-4 (model.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by steering angle offset 0.4 and -0.3 respectively. To augment training data center images were flipped (-ve steering angle). For details about the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive a car on the given track 1 without deviating the track.

Like explained in the earlier section first step was to use a convolution neural network model similar to the NVIDIA's End to End Learning for Self-Driving Cars paper. I reduced complexity in NVIDIA's by reducing the one layer i.e. 4 convolution layers instead of 5 in Nividia network. Before feeding to the first convolution layer three preprocessing steps were implemented (model.py line 76-78)
a. Cropped top portion representing sky and bottom portion representing car hood.
b. Resized image from (160, 320, 3) to (64, 64, 3)
c. Normalize the data

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80:20 split). As my model already had Maxpooling after each layer, both training and validation set had low mean squared error, validation set being lower than training set. This implied that the model was not overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. Car stayed on the track for the whole lap when driven in autonomous mode.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-98) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![CNN] [./report-images/CNN.jpg]

#### 3. Creation of the Training Set & Training Process

To train, test data provided is used, following are the images from center, right and left camera as an example.
![left-camera-image][./report-images/left_2016_12_01_13_30_48_287.jpg]
![center-camera-image][./report-images/center_2016_12_01_13_30_48_287.jpg]
![right-camera-image][./report-images/right_2016_12_01_13_30_48_287.jpg]

Steering angles for left and right images were offset by 0.4 and -0.3 degrees respectively.

To augment the data set, center camera images and steering angles were flipped. As an example, below is the center image from above being flipped.

![center-image-flipped][./report-images/center-flipped.jpg]

I finally randomly shuffled the data set and put 20% of the data into a validation set. Finally, training data set was 6428 and validation set 1608. Image processing was done in CNN for faster processing
a. Cropped top portion representing sky and bottom portion representing car hood.
b. Resized image from (160, 320, 3) to (64, 64, 3)
c. Normalize the data

Following images show left, center and right camera images being cropped in the first step of image processing

![left-camera-image-cropped][./report-images/left-cropped.jpg]
![center-camera-image-cropped][./report-images/center-cropped.jpg]
![right-camera-image-cropped][./report-images/right-cropped.jpg]

Following images show left, center and right camera images being resized to (64, 64,3) after cropping
![left-camera-image-resize][./report-images/left-resize.jpg]
![center-camera-image-resize][./report-images/center-resize.jpg]
![right-camera-image-resize][./report-images/right-resize.jpg]


The validation set helped determine if the model was over or under fitting. Ran 5 EPOCHS and validation losses kept on reducing. Adam optimizer used for force to use 1e-4 learning rate.
![training-validation-losses][./report-images/EPOCH-Losses.jpg]

Autonmous video generated using this model shows the car completing the track wihtout going off the road.
![][track1.mp4]

One curious observation in preprocessing image. I did have issues car going off the track at one particular place when I used "cv2.imread" and converted to RGB format. With everything else remaining the same if I read image using "ndimage.imread" autonomous mode was fine, and this was implemented.