import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


def MnistNetworkTeacher(input,keep_prob_conv,keep_prob_hidden,scope='Mnist',reuse = False):
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.conv2d],kernel_size = [3,3],stride = [1,1],biases_initializer=tf.constant_initializer(0.0),activation_fn=tf.nn.relu):
										
										
			net = slim.conv2d(input, 32, scope='conv1')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool1')
			net = tf.nn.dropout(net, keep_prob_conv)

			net = slim.conv2d(net, 64,scope='conv2')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool2')
			net = tf.nn.dropout(net, keep_prob_conv)

			net = slim.conv2d(net, 128,scope='conv3')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool3')
			net = tf.nn.dropout(net, keep_prob_conv)

			net = slim.flatten(net)
		with slim.arg_scope([slim.fully_connected],biases_initializer=tf.constant_initializer(0.0),activation_fn=tf.nn.relu) :
			
			net = slim.fully_connected(net,625,scope='fc1')
			net = tf.nn.dropout(net, keep_prob_hidden)
			net = slim.fully_connected(net,10,activation_fn=None,scope='fc2')
			
			net = tf.nn.softmax(net/temperature)
			return net

def MnistNetworkStudent(input,scope='Mnist',reuse = False):
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=tf.nn.sigmoid):
			
			net = slim.fully_connected(input,1000,scope = 'fc1')
			net = slim.fully_connected(net,10,activation_fn = None,scope = 'fc2')
			
			return net

def loss(prediction,output,temperature = 1):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(output * tf.log(prediction), reduction_indices=[1]))                                              
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(output,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return cross_entropy,accuracy



eps = 0.1
alpha = 0.5
temperature = 1
start_lr = 1e-4
decay = 1e-6

with tf.Graph().as_default():
		
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
	 

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	keep_prob_conv = tf.placeholder(tf.float32)
	keep_prob_hidden = tf.placeholder(tf.float32)
	x_image = tf.reshape(x, [-1,28,28,1])

	y_conv_teacher=MnistNetworkTeacher(x_image,keep_prob_conv,keep_prob_hidden,scope = 'teacher')
	y_conv = MnistNetworkStudent(x,scope = 'student')

	y_conv_student = tf.nn.softmax(y_conv/temperature)
	y_conv_student_actual = tf.nn.softmax(y_conv)

	cross_entropy_teacher,accuracy_teacher=loss(y_conv_teacher,y_,temperature = temperature)
	student_loss1,accuracy_student = loss(y_conv_student_actual,y_,temperature = temperature)
	
	student_loss2 = tf.reduce_mean( - tf.reduce_sum(y_conv_teacher * tf.log(y_conv_student), reduction_indices=1))
	cross_entropy_student = student_loss1 + student_loss2
	
	model_vars = tf.trainable_variables()
	var_teacher = [var for var in model_vars if 'teacher' in var.name]
	var_student = [var for var in model_vars if 'student' in var.name]

	grad_teacher = tf.gradients(cross_entropy_teacher,var_teacher)
	grad_student = tf.gradients(cross_entropy_student,var_student)

	l_rate = tf.placeholder(shape=[],dtype = tf.float32)
	
	trainer = tf.train.RMSPropOptimizer(learning_rate = l_rate)
	trainer1 = tf.train.GradientDescentOptimizer(0.1)

	train_step_teacher = trainer.apply_gradients(zip(grad_teacher,var_teacher))
	train_step_student = trainer1.apply_gradients(zip(grad_student,var_student))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver1 = tf.train.Saver(var_teacher)
	saver2 = tf.train.Saver(var_student)
	
	for i in range(20000):
	  batch = mnist.train.next_batch(128)
	  lr = start_lr * 1.0/(1.0 + i*decay)
	  if i%50 == 0:
		train_accuracy = accuracy_teacher.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob_conv: 1.0,keep_prob_hidden: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step_teacher.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_conv :0.8,keep_prob_hidden:0.5,l_rate:lr})
	
	saver1.save(sess,'./models/teacher1.ckpt')
	
	 
	for i in range(15000):
	  batch = mnist.train.next_batch(100)
	  if i%50 == 0:
		train_accuracy = accuracy_student.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob_conv: 1.0,keep_prob_hidden: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step_student.run(feed_dict={x: batch[0], y_: batch[1], keep_prob_conv :1.0,keep_prob_hidden:1.0})

	  
	saver2.save(sess,'./models/student.ckpt')  
	
	test_acc = sess.run(accuracy_student,feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob_conv: 1.0,keep_prob_hidden: 1.0})
	print("test accuracy of the student model is %g "%(test_acc))
		
	
	
