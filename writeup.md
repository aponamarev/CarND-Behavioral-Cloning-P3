# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_distribution.png "data_distribution"
[image2]: ./examples/original_img.jpg "original"
[image3]: ./examples/cropped_img.jpg "cropped"
[image4]: ./examples/cropped_and_flipped_img.jpg "flipped"
[video1]: ./run1.mp4 "results"

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* src/utils/general_utils.py containing utility functions
* src/Models/SimplifiedModel.py containing a description of Model architecture

model.py - provides an option for picking various model architectures. Model.h5 was generated using the following script src/Models/SimplifiedModel.py

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The model.py contains the code for training and saving the convolution neural network. As described above the model architecture was build using src/Models/SimplifiedModel.py

### Model Architecture and Training Strategy
The model was trained using the following settings:

python model.py --top_crop 50 --model_type SimplifiedModel --noshift --batch_size 256 --epochs 5 --model_name model.h5

#### 1. An appropriate model architecture has been employed

For training purpose I tested various neural networks ranging in depth between 10 to 16 of convolutional layers ranging in width between 16 to 256 channels. However, the increase in depth and width of the net architectures did not materially impact the performance of the algorithm. As a result, the following architecture was chosen:

My model consists of a 10 convolution neural network layers with 3x3 filter sizes and depths between 16 and 128 channels (src/Models/SimplifiedModel.py lines 18-31) 

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 17). The detailed model architecture is presented below:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           cropping2d_input_1         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 90, 320, 3)    0           cropping2d_1               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 90, 320, 16)   448         lambda_1               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 44, 159, 16)   2320        convolution2d_1            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 44, 159, 16)   0           convolution2d_2            
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 44, 159, 16)   64          dropout_1                  
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 44, 159, 32)   4640        batchnormalization_1       
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 21, 79, 32)    9248        convolution2d_3            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 21, 79, 32)    0           convolution2d_4            
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 21, 79, 32)    128         dropout_2                  
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 21, 79, 64)    18496       batchnormalization_2       
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 10, 39, 64)    36928       convolution2d_5            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10, 39, 64)    0           convolution2d_6            
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 10, 39, 64)    256         dropout_3                  
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 10, 39, 64)    36928       batchnormalization_3       
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 4, 19, 64)     36928       convolution2d_7            
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 4, 19, 64)     256         convolution2d_8            
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 4, 19, 64)     0           batchnormalization_4       
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 4, 19, 128)    73856       dropout_4                  
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 1, 9, 128)     147584      convolution2d_9            
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 1, 9, 128)     512         convolution2d_10           
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           batchnormalization_5       
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1152)          0           flatten_1                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           147584      dropout_5                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 64)            8256        dense_1                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            1040        dense_2                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dense_3                    
____________________________________________________________________________________________________
Total params: 525,489
Trainable params: 524,881
Non-trainable params: 608


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (src/Models/SimplifiedModel.py line 20, 24, 28, 33, and 38). 

The model was trained and validated on different subsets sets created by training and validation split (15%) to ensure that the model was not overfitting (model.py line 79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a data set provided by Udacity. The dataset alone was sufficient to train a fully functional neural net (thus eliminating a need for additional data collection). The dataset was augmented using synthetic oversampling (model.py lines 72 - 88) in order to achieve a roughly uniform distribution of steering angles.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test a lightweight nets. I started with simple one layer convolutional architecture, then compare its performance to LeNet, and the network architecture designed by NVIDIA.The best performance was demonstrated by NVIDIA net.

However, in order to reduce overfitting and increase training speed I used dropout and batch normalization. In addition, I replaced 5x5 convolutional layers with 2 convolutional layers with 3x3 kernels. The network was also modified to crop images on the top and bottom to eliminate the parts of the image that do not provide useful information. Results of crop-preprocessing presented below:

###### Original image:      | ###### Cropped image:
![original][image2] | ![cropped][image3]

At training, a selected model demonstrated 30-34% accuracy. However the accuracy was at 50-54% at validation stage. The increase in accuracy is most likely driven by a combination of simpler data samples and the effect of turning off the dropout at evaluation stage. A validation set skewed towards samples with small steering angles (driving on a straight line).The improvement in accuracy also signifies the absence of overfitting.

The final step was to run the simulator to see how well the car was driving around track one. The car successfully navigated the track one for 5 laps at default speed of 9mph. In addition, I tested the net at 15mph setting, and the model successfully completed the track.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I started with the dataset provided by Udacity. The data set was heavily skewed towards driving on the straight-line, which could result in an optimization problem. Therefore to eliminate this problem I had to rebalance the data. However, before rebalancing the dataset, I split the original set into training and validation dataset (setting 15% of the data for validation - model.py lines 61 - 67). Then, I create synthetic over-sampling algorithm that allowed me to rebalance the number of samples with high degree of steering (sharp turns) to achieve a uniform distribution.

#### The chart below presents the distribution of an original training dataset (in grey) and a rebalanced dataset (in blue):
![data_distribution][image1]

The resulting rebalanced dataset had a good mix of driving on the straight line in the middle of the road as well as going through the corners. In addition to rebalancing the data, I also randomly flipped images and angles thinking that this would remove any potential bias of oversteering in one direction.

The results of horizontal flip augmentation presented below:
#### Image before horizontal flip:
![cropped][image3]

#### Image after horizontal flip
![flipped][image4]

As a result of the preprocessing the data and splitting the data into training and validation datasets, I had:
Training set size: 34140, Validation set size: 1206

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Through experimentation I determined that the minimal necessary number of epochs was 3. However, to ensure consistency I ran training for 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The results of the algorithm are presented below:
![result][video1]
