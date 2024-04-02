#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Convolution multiple times, we'd have to make the input be a 3d numpy array.
'''

class Convolution:
    # A Convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array.
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backward(self, dL_dout, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - dL_dout is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += dL_dout[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Convolution as the first layer in our CNN.
        # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None

class MaxPool2:
    # A Max Pooling layer using a pool size of 2.

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input_data):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        self.last_input = input_data

        h, w, num_filters = input_data.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input_data):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, dL_dout):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - dL_dout is the loss gradient for this layer's outputs.
        '''
        dL_dinput = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            dL_dinput[i * 2 + i2, j * 2 + j2, f2] = dL_dout[i, j, f2]

        return dL_dinput

class Fully_Connected:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp_values = np.exp(totals)
        return exp_values / np.sum(exp_values, axis=0)

    def backward(self, dL_dout, learn_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - dL_dout is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        # We know only 1 element of dL_dout will be nonzero
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t    = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            dL_dy = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            dL_dw = d_t_d_w[np.newaxis].T @ dL_dy[np.newaxis]
            dL_db = dL_dy * d_t_d_b
            dL_dinputs = d_t_d_inputs @ dL_dy

            # Update the weights and biases based on the learning rate and gradients
            self.weights -= learn_rate * dL_dw
            self.biases  -= learn_rate * dL_db

            return dL_dinputs.reshape(self.last_input_shape)

import tensorflow.keras.datasets.fashion_mnist as fashion_mnist

# Load data from tensorflow
data = fashion_mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
plt.imshow(X_train[0])
plt.show()

# Reduce the sample size
from sklearn.model_selection import train_test_split
train_size = 10000
test_size  = 5000
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size,
                                                  test_size=test_size, shuffle=True)

print("Raw Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# Preprocess data
# Normalize data within {0,1}
X_train = X_train / 255.0
X_val   = X_val / 255.0
X_test  = X_test / 255.0

# visualizing the first 10 images in the dataset and their labels
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.title(item[y_train[i]])
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()
print('label for each of the above image: %s' % (y_train[0:20]))

epochs      = 10
lr          = 1e-3
batch_size  = 500
output_size = n_classes = len(np.unique(y_train))

# Print data shape
print("Train/Test/Validation Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# Build Model
conv = Convolution(8)                        # 28x28x1 -> 26x26x8
pool = MaxPool2()                            # 26x26x8 -> 13x13x8
softmax = Fully_Connected(13 * 13 * 8, 10)   # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    conv_out = conv.forward((image) - 0.5)
    pool_out = pool.forward(conv_out)
    out = softmax.forward(pool_out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # backward
    gradient = softmax.backward(gradient, lr)
    gradient = pool.backward(gradient)
    gradient = conv.backward(gradient, lr)

    return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(epochs):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(X_train, y_train)):
        if (i + 1) % 100 == 0:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            print("Label :\n", label )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(X_test, y_test):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(X_test)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

