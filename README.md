# Behavioral Cloning Project

---

The goal of this project is to train a deep learning model to keep a car in the lane, based on a video stream of the road as seen from a camera mounted on the car, and demonstrated driving behavior (steering angles).

The pipeline consists of the following steps:
* Use the Udacity simulator to collect data of good driving behavior
* Build a Convolutional Neural Network (CNN) in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track One in the simulator without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Lenet1.png "Model Visualization"
[image1.3]: ./unbalanced_steering_angles_hist.png "Initial distribution of steering angles"
[image1.4]: ./balanced_steering_angles_hist_1.png "Augmented distribution of steering angles"
[image1.5]: ./steering_angles_hist.png "Augmented distribution of steering angles 2"
[image2]: ./image1.jpg "Training Image"
[image3]: ./image2.jpg "Training Image"
[image4]: ./image3.jpg "Training Image"
[image5]: ./image4.jpg "Training Image"
[image6]: ./image5.jpg "Training Image"
[image7]: ./image6.jpg "Training Image"
[image8]: ./image7.jpg "Training Image"
[image9]: ./performance.png "Performance"

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

### Data Collection and Exploration

For this problem, we are ideally working on video frames captured from an actual camera mounted on a car. For this project however, data is collected using a simulator. It has two modes - a manual mode used for training, and an autonomous mode which you can use to test drive the car using your trained model. Following screenshots show what the simulator screen looks like.

[Simulator screen shots]

The training mode is like a video game in which you drive a car on a track, using the arrow keys on a keyboard or the mouse to control steering. Below is a video of the sample training run:

[training_video.mp4]

The simulator actually stores the video in the form of individual images it is comprised of. It also records the steering angle in a csv file, which is recorded against the corresponding fully qualified image file name. The file is called `driving_log.csv`, and below is a sample of this file after running the simulator in training mode.

```
C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_078.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_078.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_078.jpg,0,0,0,9.254003E-07

C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_202.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_202.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_202.jpg,0,0,0,1.584464E-06

C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_324.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_324.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_324.jpg,0,0,0,4.65583E-06
```

We see that there are actually three images for each video frame - each corresponding to the center, left or right camera mounted on the (simulated) vehicle. The last number in the row is the recorded steering angle for that frame, and that will be used as our label and ground truth. The other 3 numbers before the steering angle include speed and a couple of other fields, but we are not interested in those for now (we will use a constant speed in autonomous mode for now).

The first thing that stands out is that the steering angle is in much lower scale in the file compared to the simulator video. It turns out the video shows the angle in degrees, whereas driving_log.csv stores it in radians.

Another thing to note is that on training track 1, most of the curves are going left. This is reflected in the histogram of the steering angles from driving_log.csv:

![alt text][image1.3]

This indicates an unbalanced dataset. There are a lot more samples of left and zero steering angles, but barely any for right. This could result in the model learning to turn left all the time and may go off the road. There are a couple of strategies to account for this. A simple way is to augment the data by horizontally reflecting all images and labelling them with the negative of the corresponding steering angles. This was the approach I chose. Another approach we could use in the simulator is turn the car around on the track and record a video driving in the opposite way. Here is the histogram after augmenting the dataset with reflected images and steering angles:

![alt_text][image1.4]

The individual image frames are shaped 160x320x3. Here are some sample images from my training set:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

Note that a significant area of the upper half consisting of sky and mountain tops is irrelevant to the steering angle. We will crop out this area to get rid of irrelevant details (about 75 pixels from the top). Same is the case with the bottom almost 25 pixels (where the hood of the car can be seen). This will reduce our feature space.

### Model Selection and Architecture

Generally at this stage, we would select a few Machine Learning algorithms to experiment with, generate a few models and then select the best one. In this case, our goal is to use deep learning so we will stick to deep neural networks. If we were to choose other algorithms such as SVM or decision trees, we would also have to work on feature extraction. However deep learning has the benefit that we can completely forgo this step; the neural net layers will take care of creating relevant features out of the raw image.

To setup a working skeleton pipeline and quickly validate it end-to-end, I first started off with a single layer fully connected neural network. Once the pipeline was working, I replaced the single fully connected layer with the Lenet architecture. Choosing the architecture of neural networks is still very much a black art, so it is a good strategy to start with well-known architectures that have been proven to do well for similar problems. Lenet is probably one of the oldest and well known deep learning networks out there. It was originally developed in 1998 for recognizing handwritten digits. It is deep, but light enough to be able to run in good time on my local machine with quad cores. In another project, I have used it successfully to train a traffic sign classifier. So it was a natural first choice. The following figure shows the original architecture of Lenet:

![alt text][image1]

Here is the code from model.py that implements this architecture:

```
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D, Lambda

model = Sequential()
# Crop the input image to filter out irrelevant parts such as the sky
# and off-road portions on the sides...
model.add(Cropping2D(cropping=((75,10),(10,10)), input_shape=(160,320,3)))

# Normalize image with zero mean...
model.add(Lambda(lambda x: (x/255.0)-0.5))

# Implement Lenet architecture...
model.add(Conv2D(6, (5,5), activation='relu', strides=1, padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Conv2D(16, (5,5), activation='relu', strides=1, padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(84, activation='relu')) #64
model.add(Dropout(0.1))
model.add(Dense(1))
```
#### Layers Specification
The original Lenet architecture takes in a 32x32 grayscale image. For this project, I replaced the input layer with a 160x320x3 layer to match the input image resolutions. An alternate was to resize all input images to 32x32 (which I have done successfully in the past for traffic sign classification), but I thought we will need more details for this problem and I didn't want to lose important information. However, the input images are cropped using a Keras Cropping layer (code line 112) to filter out parts of the image that don't impact steering decision (e.g. the sky, and in this specific case parts of left and right edges, as I know we don't need to deal with other objects like cars and pederstrians appearing in these extremeties). This gives us a 75x300x3 image. The data is then normalized using a Keras lambda layer (code line 115). The first 2 layers are a pair of 2D convolution followed by a max pooling layer. The conv layer uses 6 channels, a 5x5 kernel, stride of 1, no padding and a relu activation to introduce non-linearity. We can calculate the output dimensions of this layer as follows:

```
Output Dimensions = ( (input_dims - kernel_size + 2 * padding) / stride ) + 1
Output Dim 1 =   ( (75 - 5 + 2 * 0) / 1 ) + 1 = 71
Output Dim 2 =   ( (300 - 5 + 2 * 0) / 1 ) + 1 = 296
```

The third dimension is what we chose for the Conv2D layer, in this case 6. So the output of this first Conv2D layer will be 71x296x6.

If we were using TensorFlow directly, we would need this calculation to create placeholder variables. The beauty of using Keras is that this (and much more) is automatically done for us behind the scenes. Nevertheless, it is good to know the outputs of the layers to understand what's going on.

This is followed by a max pooling layer that uses a 2x2 kernel, default stride of 2x2 and valid padding. This compacts the output dimensions to half (but not the number of channels), producing an output of 78x158x6. This is followed by another pair of convolution and max pooling, and then 3 fully connected layers of 400, 128 and 84 respectively. I added an additional layer of 400 nodes because I'm using higher resolution images, and as I said above I didn't want to lose important information. Reducing the number of nodes to 120 suddenly may cause important information to be lost. The following table summarizes all the layers comprehensively:

| Layer         		    |     Description	        			              	|  Number of weights |
|:---------------------:|:---------------------------------------------:|:------------------:|
| Input         		    | 160x320x3 RGB image   							          |                    |
| Cropping         		  | output 75x300x3 image        							    |                    |
| Normalize         		| output 75x300x3       							          |                    |
| Convolution 5x5x6     | 1x1 stride, no padding, outputs 71x296x6     	|  126k              |
| RELU					        |	Induce non-linearity   	                      |                    |
| Max pooling	      	  | 2x2 stride,  outputs 35x148x6  	              |                    |
| Convolution 5x5x16    | 1x1 stride, no padding, outputs 31x144x16   	|  71.4k             |
| RELU					        |	Induce non-linearity 			                   	|                    |
| Max pooling	      	  | 2x2 stride,  outputs 15x72x16  	            	|                    |
| Fully connected		    | Input=17280, output 400		                   	|  6.9M              |
| RELU                  | Induce non-linearity                          |                    |
| Fully connected       | Input 400, output 120                         |  80k               |
| RELU                  | Induce non-linearity                          |                    |
| Fully connected       | Input 120, output 84                          |  10k               |
|	Relu  	              |	Induce non-linearity                          |                    |
| Fully connected       | Input 84, output 1                            |  84                |
| Total Weights         |                                               |  *7.19M*           |

The original Lenet output layer has 26 nodes, which was replaced with a single linear output node since this is a regression problem and we want a steering angle as output.

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

### Simulation

Here is a video output of the car driving autonomously on track 1. [Simulation Video](./run2.mp4)

### Performance

The project was run on Dell quad-core machine with 8GB RAM. The figure below shows the performance of the deep learning network over varying sizes of training set:

![alt_text][image9]

It can be seen that the growth is linear, so Big O for this network is O(n). For every 500 samples, it takes the network about 10 secs to train. The average image size in the training data set is 14kb, so 500 samples would make 7MB. This gives about 1.4s per MB. Based on this we can calculate the run time for various sizes of big data:

1. 1 GB = 24 mins
2. 10 GB = 4 hrs
3. 100 GB = 40 hrs
4. 1 TB = 17 days

#### Performance Impact of Removing or adding layers

In this experiment, I note the runtime performance by removing the fully connected layer only, then again by removing the convolutional layer only.

| Layer         		                                 | Runtime (per 500 samples) |
|:--------------------------------------------------:|:-------------------------:|
| Lenet + 400 fully connected layer                  | 1.36 sec                  |
| Base Lenet (remove layer of 400)                   | 1.33 sec                  |
| Remove 1st convolutional (and pooling) layer       | 1.94 sec                  |

Although the fully connected layer introduces many more weights, it has a lower impact on the runtime performance of the architecture. Interestingly, the removal of the 1st convolutional layer of 6 channels has a bigger impact on performance. Firsly, it maybe a bit cryptic to see on the surface, but removal of a smaller layer of 6 channels means now we are directly connecting the input to a bigger channel of 16. So the total weights go from 71x296x6 = 126k to 71x296x*16* = 336k, which is more than the combined weights of the original 1st and second layer (126k + 71.4k = 197.4). Secondly, although this introduces fewer weights than the fully connected layer, the back-prop and optimization in convultional layers requires more processing than a single fully connected layer.

#### Performance Impact of Data Quality

This may not be so obvious, but in running the same architecture with different data sizes, there can be a difference in runtime. This could be due to variation in the optimizing of weights from one data set to another.

| Data Set       		    |     Description	        			              	      | Runtime (per 500 samples) |
|:---------------------:|:---------------------------------------------------:|:-------------------------:|
| Data Set 1         		| Better quality, accurate individual steering angles | 1.33 sec                  |
| Data Set 2            | Lower quality, many more zero steering angles       | 1.27 sec                  |

On big data size of 1 TB, this introduces a difference of about 0.7 days of training (16.1 vs 15.4 days). We could say that a difference of 1 day on this size of data should be expected due to optimization differences.

#### Performance on GPU

For this experiment, the g2.2xlarge instance was provisioned on AWS cloud, with 1 NVIDIA Kepler GK104 Grid GPU plus 8 vCPUs and 15 GB RAM. The average runtime on GPU was 0.12 sec per MB, compared to 1.4 sec per MB for quad-core. This is a 91.4% improvement in runtime! To understand the scale of difference on big data, with 1TB of data it will take 1.45 days on GPU vs 17 days on quad-core.
