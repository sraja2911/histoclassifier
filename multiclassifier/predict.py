import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))

image_path=sys.argv[1] 
img_path = os.path.join(dir_path, sys.argv[1])

img_data_list=[]

img_data_list= os.listdir(image_path)


for dataset in img_data_list:
    filename = str(img_path) + "/" + dataset
    image_size=128
    num_channels=3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('blood-bloodink-Controlmode.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, len(os.listdir('training_data')))) 
    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    #result=sess.run(y_pred, feed_dict=feed_dict_testing) prediction probabilities
    prob_result=sess.run(y_pred, feed_dict=feed_dict_testing)
    y_pred_cls=tf.argmax(prob_result, dimension=1)
    predictedclass= sess.run(y_pred_cls)
    file = open('tf_output.txt', "a+")
    file.write(dataset + ",  ")
    opt = str(predictedclass) + ",  " + str(prob_result)
    #file.write("Image Name: " + dataset + "\n" + "Prediction possiblities are (classes of the order: ")
    #file.write("Blood Class -  BloodInk - Control - Ink)"+ "\n")
    file.write(opt)
    #print >>file, opt 
    #file.write(",")
    #print >>file, predictedclass
    #file.write(result) It did not work. it writes an object
    file.close()
    