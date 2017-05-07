import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.utils.general_utils import rebalanced_set, \
    continuous_to_bins, generate_data_with_augmentation_from,\
    create_paths_to_images, ensure_valid_values
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K

from src.Models.SimplifiedModel import SimplifiedModel
from src.Models.SimplifiedModel_Extra_Dropout import SimplifiedModelExtraDropout

# Define the main parameters for the algorithm
tf.flags.DEFINE_string('data_location',
                       'data',
                       'Define the location of the data folder containing csv descriptor and IMG folder - Default: data')
tf.flags.DEFINE_string('logs_location',
                       'logs',
                       'Define the location of the logs folder. It will be used for storing models - Default: logs')
tf.flags.DEFINE_string('descriptor_name',
                       'driving_log.csv',
                       'Provide the name of the data descriptor - Default: driving_log.csv')
tf.flags.DEFINE_string('model_type',
                       'SimplifiedModelExtraDropout',
                       'Provide the name of the net architecture to be used for training [SimplifiedModel,SimplifiedModelExtraDropout] - Default: SimplifiedModel')
tf.flags.DEFINE_string('model_name',
                       'LeNet_DropOut.h5',
                       'Provide the name of the data descriptor - Default: LeNet_DropOut.h5')
tf.flags.DEFINE_integer('batch_size',256,
                        'Provide the batch size - Default: 256')
tf.flags.DEFINE_integer('epochs',5,
                        'Specify the number of epochs for the training - Default: 5')
tf.flags.DEFINE_integer('bins',5,
                        'Specify the number of bins used to rebalance the data - Default: 5')
tf.flags.DEFINE_integer('top_crop',0,
                        'Specify the number pixels to be cropped form the top - Default: 0')
tf.flags.DEFINE_integer('bottom_crop',20,
                        'Specify the number pixels to be cropped on the bottom - Default: 20')
tf.flags.DEFINE_float('val_portion', 0.15,
                      'Define the portion of the dataset used for validation')
tf.flags.DEFINE_float('shift_value', 0.20,
                      'Define the shift value for cameras - Default: 0.20')
tf.flags.DEFINE_float('width', 1.0,
                      'Define the width scaller for the net. Default: 1.0 (float)')
tf.flags.DEFINE_bool('shift', True, "Camera shift augmentation is set for True. Set for False to turn off.")
tf.flags.DEFINE_bool('flip', True, "Camera flip augmentation is set for True. Set for False to turn off.")

FLAGS = tf.flags.FLAGS
csv_file_name=FLAGS.descriptor_name
data_path = FLAGS.data_location
model_path = FLAGS.logs_location
model_name = FLAGS.model_name
batch_size = FLAGS.batch_size

input_img_shape = [160, 320, 3]

# Load a csv descriptor that contains paths to images and corresponding steering measurements.
descriptor = pd.read_csv(os.path.join(data_path, csv_file_name))

# Create a training and validation sets
if FLAGS.shift:
    # Split a provided dataset into training and validation sets.
    train_steering,val_steering, train_paths_center,val_paths, train_paths_left, _, train_paths_right, _ = \
        train_test_split(descriptor.steering, descriptor.center,
                         descriptor.left, descriptor.right, test_size=FLAGS.val_portion)
    # If the use of left and right cameras is enabled, merge concatenate paths and corresponding measurements
    # for left, center, and right cameras into a single dataset. Adjust the steering measurement accordingly.
    train_paths = np.concatenate((train_paths_left, train_paths_center, train_paths_right))
    train_steering = np.concatenate((train_steering + FLAGS.shift, train_steering, train_steering - FLAGS.shift))
else:
    # Split a provided dataset into training and validation sets.
    train_steering, val_steering, train_paths, val_paths = train_test_split(descriptor.steering, descriptor.center, test_size=FLAGS.val_portion)

# Convert partial datapaths stored in the descriptor into a full datapath that can be used for reading the data
train_paths, val_paths = create_paths_to_images(train_paths, FLAGS.data_location), create_paths_to_images(val_paths, FLAGS.data_location)

# To avoid potential bias of the data, rebalance the training dataset to ensure that the steering measurements
# are approximately uniformly distributed
# 1. Convert the continues measurements into a discrete ranges
Y_train_binned = continuous_to_bins(train_steering, n_bins=FLAGS.bins)
# 2. Create dataset indices that will represent balanced data
binned_indices = rebalanced_set(Y_train_binned)
# - Before finishing the preprocessing of the dataset, make sure that all provided measurements and images exist
train_paths, train_steering = ensure_valid_values(train_paths, train_steering)
val_paths, val_steering = ensure_valid_values(val_paths, val_steering)
# - Visualize the distribution of the data before and after rebalancing
import matplotlib.pyplot as plt
n, bins, patches = plt.hist([train_steering, train_steering[binned_indices]],
                            bins=FLAGS.bins,
                            label=['train_steering','rebalanced_steering'],
                            color=['grey', 'blue'])

plt.xlabel('Steering Values')
plt.ylabel('Samples')
plt.legend()
plt.show()
# 3. Rebalance the data by applying derived indicies to the training dataset
train_paths, train_steering = train_paths[binned_indices], train_steering[binned_indices]
# Report stats
print("Training set size: {}, Validation set size: {}".format(len(train_paths), len(val_steering)))

# Initialize the neural network
# 1. Check that a correct network architecture is requested
assert FLAGS.model_type in ['SimplifiedModel', 'SimplifiedModelExtraDropout'], \
    "Incorrect model name provided. Expected: ['SimplifiedModel', 'SimplifiedModelExtraDropout']. Provided {}".\
        format(FLAGS.model_type)
# 2. Initialize a neural network according to requested conditions
if FLAGS.model_type == 'SimplifiedModel':
    model = SimplifiedModel(FLAGS, input_img_shape)
if FLAGS.model_type == 'SimplifiedModelExtraDropout':
    model = SimplifiedModelExtraDropout(FLAGS, input_img_shape)

# By default, Tensorflow/Keras will allocated all available GPU memory, thus limiting the number of experiments that
# can be run at the same time. Therefore, in order to only allocate necessary GPU memory set gpu_options.allow_growth
# of tf.ConfigProto. Also, make sure that algorithm will automatically switch to run on CPU if GPU not availble by
# setting allow_soft_placement=True of tf.ConfigProto
config = K.tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
# Compilre resulting net to training the network using Mean Squared Error loss function and report network accuracy.
model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
# Report the resulting architecture of the net
model.summary()

# Training the network using a generator
model.fit_generator(generate_data_with_augmentation_from(train_paths, train_steering, batch_size, FLAGS.flip),
                    samples_per_epoch=len(train_steering),
                    nb_epoch=FLAGS.epochs,
                    validation_data=generate_data_with_augmentation_from(val_paths, val_steering, batch_size, False),
                    nb_val_samples=len(val_steering))

model.save(os.path.join(model_path, model_name))



