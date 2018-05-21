Binary Classifier:

A binary classes (Classes Eg: Blood, BloodInk; BloodInk, Control; Control, Ink; Blood, Control- Control: No Ink, Blood, Bubble in the images) image classifier, based on convolutional neural network using Tensorflow.

Glioblastomas (GBM) Images (SVS-WSI images are converted into jpg files) are read through opencv
(pip install opencv-python)

As a start, a very small (tutorials to histopathology images classification) network that can run on a CPU. 

Data:
Training Data: 160 Images
Validation Data: 40 Images
Test Data: 40 Images
Input Image size converted to 128 * 128 * 3  
Training images are passed in a batch of 10 (batch_size) in each iteration (16 iterations).

Network architecture
RELU as our activation function which simply takes the output of max_pool and applied RELU using tf.nn.relu. All these operations are done in a single convolution layer.

Used k_size/filter_size as 2 * 2 and stride of 2 in both x and y direction.

Used AdamOptimizer for gradient calculation and weight optimization. Tried to  minimise cost with a learning rate of 0.0001.

-----------------------------------------------------------------------------------------------------------
Multiclassifer:

A multi classes (Classes Blood, BloodInk, Control, Ink; Control: No Ink, Blood, Bubble in the images) image classifier, based on convolutional neural network using Tensorflow.

Glioblastomas (GBM) Images (SVS-WSI images are converted into jpg files) are read through opencv (pip install opencv-python)

Data: 
Training Data: 320 Images,  Validation Data: 80 Images,  Test Data: 80 Images
Input Image size converted to 128 * 128 * 3
Training images are passed in a batch of 10 (batch_size) in each iteration (32 iterations).

Network architecture RELU as our activation function which simply takes the output of max_pool and applied RELU using tf.nn.relu. All these operations are done in a single convolution layer.

Used k_size/filter_size as 2 * 2 and stride of 2 in both x and y direction.

Used AdamOptimizer for gradient calculation and weight optimization. Tried to minimise cost with a learning rate of 0.00001.

Training Accuracy: 40-70% and Validation Accuratcy: 0 - 80%

WIP Docs folder has runtime results in a doc file with metrices in a spreadsheet file

Prediction Metrices for each class:

Classess in the order: Blood, BloodInk, Control, Ink     

For Blood   : [[0.41105273, 0.10850155 , 0.250138, 0.23030767]]     

For BlookInk: [[0.47074214, 0.07581728, 0.2964673, 0.1569733 ]] 

For Control : [[0.10678441, 0.50450104, 0.13086669, 0.25784788]]                                                             

For Ink     : [[0.08805854, 0.2579561,  0.31059676 0.343388]]
