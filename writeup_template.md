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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
The model.py contains the code for training and saving the convolution neural network. As described above the model architecture was buidl using src/Models/SimplifiedModel.py

### Model Architecture and Training Strategy
The model was trained using the following settings:

python model.py --data_location data/attempt2 --shift_value 0.2 --batch_size 128 --top_crop 50 ----batch_size 128 epochs 5 --model_name SimpleModel_attempt2_shift_0.2.h5

#### 1. An appropriate model architecture has been employed

For training purpose I tested various neural networks ranging in depth between 10 to 16 of convolutional layers ranging in width between 16 to 256 channels. However, the increase in depth and width of the net architectures did not materially impact the performance of the algorithm. As a result, the following architecture was chosen:

My model consists of a 10 convolution neural network layers with 3x3 filter sizes and depths between 16 and 128 channels (src/Models/SimplifiedModel.py lines 18-31) 

The model includes ELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 17). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (src/Models/SimplifiedModel.py line 34). 

The model was trained and validated on different subsets sets created by training and validation split (15%) to ensure that the model was not overfitting (model.py line 79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, sharp corner steering, and left and right cameras. The created dataset was normalized to achieve a uniform distribution of steering angles (in the ground truth).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test a simple one layer convolutional architecture, then compare its performance to LeNet, and the network architecture designed by NVIDIA.

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
