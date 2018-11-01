# Behavioral Cloning Project

---

This project implements Behavioral Cloning for a self-driving car using deep neural networks with Keras and Tensorflow backend. The result is a model that learns to keep a car in the lane, based on a video stream of the road as seen from a camera mounted on the car, and demonstrated driving behavior in the form of steering angles.

The pipeline consists of the following steps:
* Use a simulator to collect data of good driving behavior
* Build a Convolutional Neural Network (CNN) in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track One in the simulator without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image0.1]: ./images/sim1.PNG "Simulator"
[image0.2]: ./images/sim2.PNG "Simulator"
[image1]: ./images/Lenet1.png "Model Visualization"
[image1.3]: ./images/unbal_steering_angles_hist.png "Initial distribution of steering angles"
[image1.4]: ./images/balanced_steering_angles_hist_1.png "Augmented distribution of steering angles"
[image1.5]: ./images/steering_angles_hist.png "Augmented distribution of steering angles 2"
[image2]: ./images/image1.jpg "Training Image"
[image3]: ./images/image2.jpg "Training Image"
[image4]: ./images/image3.jpg "Training Image"
[image5]: ./images/image4.jpg "Training Image"
[image6]: ./images/image5.jpg "Training Image"
[image7]: ./images/image6.jpg "Training Image"
[image8]: ./images/image7.jpg "Training Image"
[image9]: ./images/performance.png "Performance"

### Table of Contents:

* [Data Collection and Exploration](#data-collection-and-exploration)
* [Model Selection and Architecture](#model-selection-and-architecture)
  * [Layers Specification](#layers-specification)
  * [Error Function](#error-function)
* [Training Strategy, Tuning and Results](#training-strategy-tuning-and-results)
  * [Handling Overfitting](#handling-overfitting)
  * [Model Parameter Tuning](#model-parameter_tuning)
* [Simulation](#simulation)
* [Performance](#performance)
  * [Performance on GPU on AWS](#performance-on-gpu-on-amazon-cloud)
  * [Performance Impact of Removing and Adding Layers](#performance-impact-of-removing-or-adding-layers)

---

### Data Collection and Exploration

For this project implementation, data is collected using a simulator. It has two modes - a manual mode used for training, and an autonomous mode which you can use to test drive the car using your trained model. Following screenshots show what the simulator screen looks like.

![alt text][image0.1]

Clicking on Training Mode takes us to a screen like this:

![alt text][image0.2]

The training mode is like a video game in which you drive a car on a track, using the arrow keys on a keyboard or the mouse to control steering. Below is a video of the sample training run:

[Training Video](./videos/sample_training_video.mp4)

The simulator actually stores the video in the form of individual images it is comprised of. It also outputs `driving_log.csv` which records the steering angle against the corresponding fully qualified image file name. Below is a sample of this file after running the simulator in training mode.

```
C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_078.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_078.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_078.jpg,0,0,0,9.254003E-07

C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_202.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_202.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_202.jpg,0,0,0,1.584464E-06

C:\ahmed\CarND\behavioral_cloning\IMG\center_2018_07_30_19_54_48_324.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\left_2018_07_30_19_54_48_324.jpg,C:\ahmed\CarND\behavioral_cloning\IMG\right_2018_07_30_19_54_48_324.jpg,0,0,0,4.65583E-06
```

We see that there are actually three images for each video frame - each corresponding to the center, left or right camera mounted on the (simulated) vehicle. The last number in the row is the recorded steering angle for that frame, and that will be used as our label and ground truth. The other 3 numbers before the steering angle include speed and a couple of other fields, but we are not interested in those for now (we will use a constant speed in autonomous mode for now).

The first thing that stands out is that the steering angle in the file is in radians, compared to degrees in the video.

The individual image frames are shaped 160x320x3. Here are some sample images from my training set:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

Note that a significant area of the upper half consisting of sky, tree tops and hills is irrelevant to the steering angle. We will crop out this area to get rid of irrelevant details (about 75 pixels from the top). Same is the case with the bottom almost 25 pixels (where the hood of the car can be seen). This will reduce our feature space.

Another thing to note is that on training track 1, most of the curves are going left. This can be seen in the histogram of the steering angles from driving_log.csv:

![alt text][image1.3]

This indicates an unbalanced dataset. This could result in the model learning to turn left all the time and may go off the road. There are a couple of strategies to account for this. A simple way is to augment the data by horizontally reflecting all images and labelling them with the negative of the corresponding steering angles. This was the approach chosen here. Another approach we could use in the simulator is to turn the car around on the track and record a video driving in the opposite way. 

Here is the histogram after augmenting the dataset with reflected images and steering angles:

![alt_text][image1.4]

There is another remaining issue. Note that in the histogram above, the angles are not normally distributed. Because this is a regression problem, we intend to use the Root Mean Squared Error (RMSE) based error optimization scheme. RMSE assumes a normal distribution, so it will not be effective if data is not normally distributed. 

This is actually a result of how data was collected. The training lap used keyboard controls, which causes acute steering changes. This introduces large variances in training data and as a result the network can't learn the subtle steering angles that are needed to keep the car on track. This is indicative of the importance of data collection process, and an example of how errors and noise can be introduced due to issues in that process.

To collect good data, I stopped the car at various points on the track, specifically set an appropriate steering angle with mouse (keyboard doesn't allow that precision) for that location and pose of the car on the track, and recorded a quick shot of the pose by starting and stopping recording immediately. I took these snapshots in the middle of the road, as well as on the side lanes (to train for recovery). In some places, I drove the car at a very low speed and recorded a short window. I did this while driving straight in the center, as well as during recovery from side lines. This resulted in about 3000 center camera images. Here is a histogram showing the final distribution of steering angles in the training data (after horizontal reflection):

![alt text][image1.5]

The data still has many more straight (zero) angles, but it is normally distributed and smoother than the original one. 

### Model Selection and Architecture

Generally at this stage, we would select a few Machine Learning algorithms to experiment with, generate a few models and then select the best one. In this case, our goal is to use deep learning so we will stick to deep neural networks. If we were to choose other algorithms such as SVM or decision trees, we would also have to work on feature extraction. However deep learning has the benefit that we can completely forgo this step; the neural net layers will take care of creating relevant features out of the raw image.

To setup a working skeleton pipeline and quickly validate it end-to-end, I first started off with a single layer fully connected neural network. Once the pipeline was working, I replaced the single fully connected layer with the Lenet architecture. Choosing the architecture of neural networks is still very much an emperical process, so a good strategy is to start with well-known architectures that have been proven to do well for similar problems. Lenet is probably one of the oldest and well known deep learning networks out there. It was originally developed in 1998 for recognizing handwritten digits. It is deep, but light enough to be able to run in good time on my local machine with quad cores. In another project, I have used it successfully to train a traffic sign classifier. So it was a natural first choice. The following figure shows the original architecture of Lenet:

![alt text][image1]

#### Layers Specification
The original Lenet architecture takes in a 32x32 grayscale image. For this project, the original input layer is replaced with a 160x320x3 layer to match the input image resolutions. An alternate is to resize all input images to 32x32, but it looks though we will need more details for this problem and I didn't want to lose important information due to resizing. However, the input images are cropped using a Keras Cropping layer to filter out parts of the image that don't impact steering decision (e.g. the sky, and parts of left and right edges as I know we don't need to deal with other objects like cars and pederstrians appearing in these extremeties). This gives us a 75x300x3 image. The data is then normalized using a Keras lambda layer. Here we can use a straightforward normalization based on domain knowedge (we know image pixels can only range from 0 to 255), but in other cases we would do this based on the mean and standard deviation of the dataset, or use a library like StandardScaler from sklearn. 

The first 2 layers are a pair of 2D convolution followed by a max pooling layer. The conv layer uses 6 channels, a 5x5 kernel, stride of 1, no padding and a relu activation to introduce non-linearity. We can calculate the output dimensions of this layer as follows:

```
Output Dimensions = ( (input_dims - kernel_size + 2 * padding) / stride ) + 1
Output Dim 1 =   ( (75 - 5 + 2 * 0) / 1 ) + 1 = 71
Output Dim 2 =   ( (300 - 5 + 2 * 0) / 1 ) + 1 = 296
```

The third dimension is the number of channels we chose for the convolution output, in this case 6. So the output of this first Conv2D layer will be 71x296x6.

If we were using TensorFlow directly, we would need this calculation to create placeholder variables. The beauty of using Keras is that this, and much more, is automatically done for us behind the scenes. Nevertheless, it is good to know the outputs of the layers to understand what's going on.

The RELU (or Rectified Linear Unit) layer has become the method of choice to introduce non-linearity in deep networks. Compared to the traditional sigmoid or tanh functions, RELU address the problem of the vanishing gradient - which is more common in deep networks with many more layers and epochs. Basically, it suppresses any negative output by setting it to zero while outputting a linear function for a positive output. Non-linearity is key to learning complex functions in neural networks.

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

#### Error Function
The loss function used is mean-squared error, whereas the Adam Optimizer was used to minimize it (more on this below).

```
 model.compile(loss='mse', optimizer='adam')
```

### Training Strategy, Tuning and Results

The above architecture trained on the training data had a very decent validation loss, and did pretty well even when trained for only 2 epochs, as can be seen below:
 
```
Train on 2972 samples, validate on 744 samples
Epoch 1/2
2972/2972 [==============================] - 132s 44ms/step - loss: 0.0613 - val_loss: 0.0713
Epoch 2/2
2972/2972 [==============================] - 122s 41ms/step - loss: 0.0095 - val_loss: 0.0410
```
There were a couple of instances of going off-road. That was corrected by recording specific poses for those spots, and training the previously saved model for one more epoch with the added training data. The validation loss went further down:
```
Train on 2972 samples, validate on 744 samples
Epoch 1/1
2972/2972 [==============================] - 130s 44ms/step - loss: 0.0042 - val_loss: 0.0318
```
This finally produced a model that did a complete lap on track 1 without going off-road.

#### Handling Overfitting

The model used 80/20 train-validate split to ensure that the model did not overfit. Dropout layers were added in order to reduce overfitting. Dropout is a newer technique to avoid overfitting in deep networks, and is both conceptually and practically simpler to implement than regularization. Conventional regularization works by adding a penalty term to the error measure, which accentuates larger model weights, thereby reducing them in subsequent training iterations. The penalty term itself is also weighted by yet another hyper-parameter. Instead, in the dropout scheme, we randomly drop some connections between 2 layers of the network. It seems a bit strange at first, but in reality works quite well. What it does is forces the network to learn redundant representations of the important features. This is specially true for deep networks as there is a large number of connections between layers. In regularization, it is also possible for some features to go down to zero (particularly in L1 regularization), so theoretically the concept is similar.

You have to decide the percentage of dropout. Starting with 50% is a frequent rule of thumb. I experimented with higher and lower dropout rates, and finally settled on 20% dropout after the first two fully connected layers of Lenet, and a 10% dropout after the last fully connected layer.

#### Model parameter tuning

Traditionally, we would also spend quite some time on hyper-parameter tuning, specially if we use gradient descent or stochastic gradient descent. A couple of the key hyper parameters are learning rate and momentum. The basic idea is that with each iteration, the weights are updated only by a small amount specified by the learning rate, otherwise it may cause huge swings in weight updates and cause the weights to oscillate between one and the other extreme. Momentum helps in avoiding the model getting stuck in a local minimum. This model used the Adam Optimizer, which adjusts the learning rates for each weight over iterations by using the first and second moments (mean and variance) of the gradients. This way it provides an adaptive process for the weight updates, so we don't have to tune the learning rate and momentum explicitly.

### Simulation

Here is a video output of the car driving autonomously on track 1. [Simulation Video](./videos/run2_autonomous.mp4)

### Performance

The project was run on Dell quad-core machine with 8GB RAM. The figure below shows the performance of the deep learning network over varying sizes of training set:

![alt_text][image9]

It can be seen that the growth is linear, so Big O for this network is O(n). For every 500 samples, it takes the network about 10 secs to train. The average image size in the training data set is 14kb, so 500 samples would make 7MB. This gives about 1.4s per MB. Based on this we can calculate the run time for various sizes of big data:

1. 1 GB = 24 mins
2. 10 GB = 4 hrs
3. 100 GB = 40 hrs
4. 1 TB = 17 days

#### Performance on GPU on Amazon Cloud

For this experiment, a g2.2xlarge instance was provisioned on AWS cloud, with 1 NVIDIA Kepler GK104 Grid GPU plus 8 vCPUs and 15 GB RAM. The average training runtime on GPU was 0.12 sec per MB, compared to 1.4 sec per MB for quad-core. This is a 91.4% improvement in runtime! To understand the scale of difference on big data, with 1TB of data it will take 1.45 days on GPU vs 17 days on quad-core to train the model.

For training this model on the cloud, I had to generate and upload an 81MB zip file of my training images data. This data size is small enough to be feasible for multipe iterations of fresh data collection, upload and training. But for big data (on the order of GBs or TBs), this could be a significant challenge.

Many popular development environments are still not optimized for cloud development. You generally need 2-3 parallel channels for this type of environment. 

1. A channel for transferring files back and forth from your server or storage in the cloud, e.g. in this case uploading my training data, and downloading my trained model.
1. A channel for running your program on the cloud, e.g. in this case my training program, model.py.
1. Optionally, a separate channel for quick edits to your code and checking into your repository directly from cloud. For bigger changes, you are probably better off editing your changes in your local repository, committing them and downloading on your cloud instance.

I find it useful to open 3 separate shells for this purpose. Two connected to my cloud instance, and one for my local file system (for uploading and downloading files). In an ideal cloud IDE, these functions would be provided by the File or Project Explorer pane, the Code Editor pane, and the Output pane.

#### Performance Impact of Removing or adding layers

In this experiment, I note the runtime performance by removing the fully connected layer only, then again by removing the convolutional layer only. This experiment was run on quad-core CPUs.

| Layer         		                                 | Runtime (per 500 samples) |
|:--------------------------------------------------:|:-------------------------:|
| Lenet + 400 fully connected layer                  | 1.36 sec                  |
| Base Lenet (remove layer of 400)                   | 1.33 sec                  |
| Remove 1st convolutional (and pooling) layer       | 1.94 sec                  |

Although the fully connected layer introduces many more weights, it has a lower impact on the runtime performance of the architecture. Interestingly, the removal of the 1st convolutional layer of 6 channels has a bigger impact on performance. Firsly, it maybe a bit cryptic to see on the surface, but removal of a smaller layer of 6 channels means now we are directly connecting the input to a bigger channel of 16. So the total weights go from 71x296x6 = 126k to 71x296x*16* = 336k, which is more than the combined weights of the original 1st and second layer (126k + 71.4k = 197.4). Secondly, although the convolutional layer introduces fewer *weights* than the fully connected layer, the kernel still has to be convolved over the entire image, which is a costly operation. We are convolving 5x5x16 cells for each of the 71x296 cells. So the total operations are 5x5x6x71x296 = 3.1M. This is still about half of the number of operations for the first fully connected layer. The other aspects to consider to explain this would be whether tensorflow vectorizes convolutions, and the overhead of calculating backprop for convolutional layers. An in-depth analysis of these aspects is beyond the scope of this report.

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
