import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


def MnistNetwork(input,scope='Mnist',reuse = False):
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=tf.nn.sigmoid):
			
			net = slim.fully_connected(input,1000,scope = 'fc1')
			net = slim.fully_connected(net,10,activation_fn = None,scope = 'fc2')
			net = tf.nn.softmax(net)
			return net


def loss(prediction,output):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(output * tf.log(prediction), reduction_indices=[1]))
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return cross_entropy,accuracy


with tf.Graph().as_default():
		
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	y_conv=MnistNetwork(x)


	cross_entropy,accuracy=loss(y_conv,y_)	
	
	trainer = tf.train.GradientDescentOptimizer(0.1)
	
	train_step = trainer.minimize(cross_entropy)

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	
	
	for i in range(15000):
	  batch = mnist.train.next_batch(100)
	  if i%50 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

	
	

	test_acc = sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
	print("test accuracy is %g"%(test_acc))
		
	
	
