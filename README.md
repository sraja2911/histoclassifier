# Multi-classes Tensorflow images classifier for Histopathology images of TCGA-GBM WSI images

A multi classes (Markers in the WSI images are classified as  Blood, BloodInk, Control, Ink; Control: No Ink, Blood, Bubble in the images) image classifier, based on convolutional neural network using Tensorflow. A multi-label classifier (having one fully-connected layer at the end), with multi-classification (four classes, in this instance).

Glioblastomas (GBM) Images (SVS-WSI images, downloaded from TCGA) are converted into jpg files) and are read through opencv (pip install opencv-python)

Original WSI-SVS images are of size : 18500 * 12000 pixels
converted JPG images are of size: 1000 * 850 pixels

320 training images (80 from each classes) are fed through 32 interations of batch size =10
20% (64 images) images automatically taken out as validation imaes, Test Data: 80 Images 

# CNN Layers
While training, all the images from four classes are fed to a convolutional layer which is followed by 2 more convolutional layers (filter size:3*3,#of filters=32). 
After convolutional layers, flattened the output and added two fully connected layer in the end. 
The second fully connected layer (128 Neurons) has only four outputs which represent the probability of an image being a Blood, BloodInk, Control or Ink classes. 

# Tensorflow Placeholders
All the input images are resized to 128 x 128 x 3 size. Input placeholder x is created in the shape of [32, 128, 128, 3]. A placeholder  (y_true) to store prediction is created. y_pred is the placeholder for output probabilities (for batch size 32 of 4 classes, it will be [32 4]).

Softmax is a function that converts K-dimensional vector ‘x’ containing real values to the same shaped vector of real values in the range of (0,1), whose sum is 1. Applied the softmax function to the output of convolutional neural network to convert the output to the probability for each class.

# Network architecture:
RELU as our activation function which simply takes the output of max_pool and applied RELU using tf.nn.relu. 
All these operations are done in a single convolution layer.

Used k_size/filter_size as 2 * 2 and stride of 2 in both x and y direction.

AdamOptimizer for gradient calculation and weight optimization. 

Tried to minimise cost with a learning rate from 0.00001 to 0.00001.
Training Accuracy: 40-70% and Validation Accuratcy: 0-80% (120 Epochs)

Tensorflow session object is used to store all the values. 

Ttraining images with labels are used for training, in general training accuracy will be higher than validation. 

After each Epooch, training accuracy and validation accuracy numbers are reported and saved the model using saver object in Tensorflow.

# Output
WIP Docs folder has runtime results
Output file (tf_output.txt) listout, probabilities of each class against each image (80 unlabeled images), with higher probabilites is considered for the said image.
