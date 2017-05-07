import numpy as np
import cv2, os
from sklearn.utils import shuffle

# Defina a utility function that will convert the partial path to the absolute one
create_paths_to_images = lambda x, data_path: np.array([os.path.join(data_path, v) for v in x])
# Define a utility function that will read and convert an image into RGB color scheme
read_rgb_img = lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def ensure_valid_values(paths, measure, dtype=np.float64):
    # Convert input into a np.array
    paths = np.array(paths)
    measure = np.array(measure)
    # Create a placeholders for new values
    new_paths, new_measure = [],[]

    for p, m in zip(paths, measure):
        # Check whether the provided path exists and whether the provided measurement of an appropriate type
        if os.path.exists(p) and (type(m)==dtype):
            # If provided path and measurement are correct, add these items to the lists that will be returned
            new_paths.append(p), new_measure.append(m)
        else:
            # If the data is incorrect, report both the path and the measurement
            print("Incorrect path:", p, m, type(m).__name__)
    # Convert results into numpy arrays.
    new_paths = np.array(new_paths)
    new_measure = np.array(new_measure)
    # Make sure that the results are not empty
    assert len(new_paths)>0, "Provided incorrect paths. No paths or measure will be generated."

    return new_paths, new_measure

def continuous_to_bins(vector, n_bins=9):
    """
    Function converts a continues input into a discrete output representing n ranges.
    :param vector: Continues input that will be converted into a discrete output
    :param n_bins: number of ranges
    :return: discrete outputs representing ranges
    """
    # Convert input into a np.array
    vector = np.array(vector)
    # Store the shape of the vector in order to reshape your results accordingly
    shape = vector.shape
    # Flatten the vector
    vector = np.reshape(vector, [-1])
    # Evalute the range of the vector and the size of each range bin
    range = vector.max()-vector.min()
    step = range/float(n_bins-1)
    # Transform continues values into discrete ranges
    binned = np.round((vector - vector.min())/step, 0)
    # Reshape results into the original shape
    binned = np.reshape(binned, shape)
    # Ensure that resulting format is integer. This step is necessary to use the results as indices
    return binned.astype(int)


def rebalanced_set(labels):
    """
    Function analyzes the dataset labels and returns indices that will rebalance the dataset
    :param labels: 
    :return: index of rebalanced dataset
    """

    labels = np.array(labels)

    # Evaluate the number of samples of each label (class) in the dataset
    # 1. Aggregate all the elements into classes
    # 1.1 Create a 2d placeholder that will hold all elements of each class separately
    train_class_indexes = [[] for _ in range(labels.max() + 1)]
    n_samples = np.zeros(labels.max() + 1)
    for i, l in enumerate(labels):
        # Sort the elements into class bins
        train_class_indexes[l].append(i)
        n_samples[l] += 1

    # Oversample the bins that are under-represented in the dataset
    for i, l in enumerate(train_class_indexes):
        size = len(l)
        for _ in range(int(n_samples.max()) - size):
            train_class_indexes[i].append(l[np.random.randint(size)])

    # Flatten out the resulting array
    train_class_indexes = np.reshape(np.array(train_class_indexes, dtype=np.int), [-1])

    # Make sure that the indices are not groupped into homogeneous series (shuffle)
    train_class_indexes = shuffle(train_class_indexes)

    return train_class_indexes



def generate_data_with_augmentation_from(paths,
                                         measurements,
                                         batch_sz=32,
                                         random_flip=True):

    # Evaluate the size of the dataset
    epoch_sz = len(measurements)

    # Define utility methods for horizontal flip of an image and for keeping image without any changes
    # These mothods will be randomly selected to decided whether perform the horizontal flip of not
    horizontal_flip = lambda x, m: (cv2.flip(x, 0), -m)
    no_flip = lambda x, m: (x, m)

    # Define an augmentation function that will be used on all images
    def augmentation(p,m):
        # Read images and convert into RBG color scheme
        im, m = read_rgb_img(p), m
        if random_flip:
            # Randomly chose between horizontal flip and no changes options
            flip_choice = np.random.choice([horizontal_flip, no_flip])
            # execute the transformation
            im, m = flip_choice(im, m)

        return im, m

    # Lunch the infinite loop that will be exected off the main tread to generate new samples
    while True:
        # For each new epoch shuffle the data
        paths, measurements = shuffle(paths, measurements)
        # Choose whether to deliver results in a batch or not
        if batch_sz is None:
            # If the batch size is not set, provide samples one by one
            # Extract path and measurement of the next sample in the loop
            for p, m in zip(paths, measurements):
                # Read and augment the image
                im, m = augmentation(p, m)
                yield (im, m)
        else:
            # If the batch size provided, process samples in a minibatch
            # Define a range for the minibatch in a loop
            for start, end in zip(range(0, epoch_sz, batch_sz),
                                  range(batch_sz, epoch_sz + 1, batch_sz)):
                # Create placeholders that will accumulate samples of the minibatch
                ims, ms = [],[]
                # Extract path and measurement of the next sample in the loop
                for p,m in zip(paths[start:end], measurements[start:end]):
                    # Read and augment the image
                    im, m = augmentation(p,m)
                    # Add resulting samples to the minibatch placeholders
                    ims.append(im)
                    ms.append(m)
                # Return resulting minibatches as an numpy array
                yield (np.array(ims), np.array(ms))

