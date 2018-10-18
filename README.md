# Behavioral Cloning Project

---

The goal of this project is to train a deep learning model to keep a car in the lane, based on demonstrated driving behavior (steering angles) and a video stream of the road as seen from a camera mounted on the car.

The pipeline consists of the following steps:
* Use the Udacity simulator to collect data of good driving behavior
* Build a Convolutional Neural Network (CNN) in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track One in the simulator without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Lenet1.png "Model Visualization"
[image1.5]: ./steering_angles_hist.png "Distribution of Steering Angles"
[image2]: ./image1.jpg "Training Image"
[image3]: ./image2.jpg "Training Image"
[image4]: ./image3.jpg "Training Image"
[image5]: ./image4.jpg "Training Image"
[image6]: ./image5.jpg "Training Image"
[image7]: ./image6.jpg "Training Image"
[image8]: ./image7.jpg "Training Image"

---
#### Files Submitted

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### How to simulate 
Using the Udacity provided simulator, drive.py and my model file model.h5, the car can be driven autonomously around the track by executing: 

```sh
python drive.py model.h5
```

### Architecture

To setup a skeleton pipeline, and quickly make sure everything is working end-to-end, I first started off with a single layer fully connected network. Once the pipeline was working, I replaced the single fully connected layer with the Lenet architecture (model.py lines 118-129) - since I had some experience with it in the traffic sign classifier project. It is deep, but light enough to be able to run in good time on my local machine with quad cores. The following figure shows the architecture of Lenet:

![alt text][image1]

#### Layers
The original Lenet architecture takes in a 32x32 image. For this project, I replaced the input layer with a 160x320x3 layer to match the input image resolutions. The input images are cropped using a Keras cropping layer (code line 112), and the data is normalized using a Keras lambda layer (code line 115). The first 2 layers are a pair of 2D convolution followed by a max pooling layer. The conv layer uses a 5x5 kernel, stride of 1, valid padding and a relu activation to introduce non-linearity. This produces 6 channels of output of 28x28 pixels. This is followed by a max pooling layer that uses a 2x2 kernel, default stride of 2x2 and valid padding. This is followed by another pair of conv2D and maxpool layer, with The original Lenet output layer has 26 nodes, which was replaced with a single linear output node since this is a regression problem and we want a steering angle as output.

#### Error Function
The loss function used was mean-squared error, whereas the Adam Optimizer was used to minimize it, which avoids the need to explicitly tune a learning rate.

I also tried more complex networks, first by just experimenting with higher number of channels in the convolution layers and a denser fully connected layer, then trying the NVIDIA network as well. But I didn't see a significant improvement over Lenet.

The initial training and validation loss was high. I re-ran my simulator for training data, trying to record better steering angles, and turning the recording on/off at appropriate times. This helped improve the training loss, but the validation loss was still high. So I introduced dropout layers. I started off with a high dropout rate of 0.5, but the validation loss was stuck at a high number through the epochs. I thought maybe I am dropping too many connections, so I lowered the rate. Experimenting this way, I got slightly better validation loss.

I tried my model on the driving track throughout the process. It was consistently veering off the road quite often. And seemingly my recovery training effort was not working either. The main cause of problems, and the most time consuming part for me was a trivial thing - I was using cv2 to load images, which uses BGR format, and the Udacity provided drive.py used PIL (which uses the RGB format). Once I figured that out, it was a breeze from there.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Handling Overfitting

I added dropout layers in order to reduce overfitting (model.py lines 124-128). I experimented with higher and lower dropout rates, and finally settled on 20% dropout after first two fully connected layers of Lenet, and a 10% dropout after the last fully connected layer.

The model used 80/20 train-validate split to ensure that the model did not overfit (model.py line 141). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used the Adam Optimizer, so the learning rate was not tuned manually (model.py line 141).

### 4. Training Strategy

Initially I recorded a couple of continuous driving laps using the arrow keys. I split this data into 80/20 for training/validation. I used random shuffling. Although I got decent validation loss, the car in autonomous mode kept veering off road at different points. Then I took a different approach. What I did was stopped the car at various points on the track, specifically set an appropriate steering angle with mouse for that location and pose of the car on the track, and recorded a quick shot of the pose by starting and stopping recording immediately. Little issues like using keyboard vs mouse control was important for this training data, because each behaves differently enough to make a difference in recording the correct training labels. I took these snapshots in the middle of the road, as well as on the side lanes (to train for recovery). In some places, I drove the car at a very low speed and recorded a short window. I did this while driving straight in the center, as well as during recovery from side lines. I recorded about 3000 center camera images in this fashion. Then I further augmented this data set using horizontal reflection to balance turning angles, as the track is mostly turning left. Here is a histogram showing the final distribution of steering angles in the training data:

![alt text][image1.5]

The data still has many more straight (zero steering angle) and slight left steering. That's because the track is mostly left or straight. Nevertheless, it is normally distributed around the mean. The model trained on this data had a pretty decent validation loss, and did pretty well even when trained for only 2 epochs, as can be seen below:
```
Train on 2972 samples, validate on 744 samples
Epoch 1/2
2972/2972 [==============================] - 132s 44ms/step - loss: 0.0613 - val_loss: 0.0713
Epoch 2/2
2972/2972 [==============================] - 122s 41ms/step - loss: 0.0095 - val_loss: 0.0410
```
There were a couple of instances of going off-road, but that was corrected by recording specific poses for those spots, and training the previously saved model for one more epoch with the added training data. The validation loss went further down:
```
Train on 2972 samples, validate on 744 samples
Epoch 1/1
2972/2972 [==============================] - 130s 44ms/step - loss: 0.0042 - val_loss: 0.0318
```
This finally produced a model that did a complete lap on track 1 without going off-road.

Here are some sample images from my training set:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]


### Simulation

Here is a video output of the car driving autonomously on track 1. [Simulation Video](./run2.mp4)
