A dual classes (Classes Eg: Blood, BloodInk; BloodInk, Control; Control, Ink; Blood, Control- Control: No Ink, Blood, Bubble in the images) image classifier, based on convolutional neural network using Tensorflow.

Glioblastomas (GBM) Images (SVS-WSI images are converted into jpg files) are read through opencv
(pip install opencv-python)

As a start, a very small (tutorials to histopathology images classification) network that can run on a CPU. 

Data:
Training Data: 100 Images
Validation Data: 20 Images
Test Data: 20 Images
Input Image size converted to 128*128*3  
Training images are passed in a batch of 20 (batch_size) in each iteration (5 in this case).

Network architecture
RELU as our activation function which simply takes the output of max_pool and applied RELU using tf.nn.relu. All these operations are done in a single convolution layer.

Used k_size/filter_size as 2*2 and stride of 2 in both x and y direction.

Used AdamOptimizer for gradient calculation and weight optimization. Tried to  minimise cost with a learning rate of 0.0001.
