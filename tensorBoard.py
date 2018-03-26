#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:42:41 2018

@author: zhao
"""

import tensorflow as tf
import input_data
max_steps=500
learning_rate=0.001
dropout=0.9
data_dir='/tmp/MNIST_data'
log_dir='/tmp/tensorflow/mnist/logs/mnist_with_summaries'

mnist=input_data.read_data_sets(data_dir,one_hot=True)
sess=tf.InteractiveSession()

with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y_=tf.placeholder(tf.float32,[None,10],name='y-input')
    
with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)                    #summary

#the function to initial the weight and bias  
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#the function to variable summary
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

#the function to build layer(includeing summary)
def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights=weight_variable([input_dim,output_dim])
            variable_summaries(weights)                               #summary
        with tf.name_scope('biases'):
            biases=bias_variable([output_dim])
            variable_summaries(biases)                                #summary
        with tf.name_scope('Wx_plus_b'):
            preactivate=tf.matmul(input_tensor,weights)+biases
            tf.summary.histogram('pre_activations',preactivate)       #summary
        activations=act(preactivate,name='activation')
        tf.summary.histogram('activations',activations)               #summary
        return activations
    
hidden1=nn_layer(x,784,500,'layer1')
with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability',keep_prob)           #summary
    dropped=tf.nn.dropout(hidden1,keep_prob)
y=nn_layer(dropped,500,10,'layer2',act=tf.identity)

#loss
with tf.name_scope('cross_entropy'):
    diff=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    with tf.name_scope('total'):
        cross_entropy=tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)                      #summary

#train_step and accuracy
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)                                #summary


merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter(log_dir+'/train',sess.graph)
test_writer=tf.summary.FileWriter(log_dir+'/test')
tf.global_variables_initializer().run()

#the function to get data and keep_prob
def feed_dict(train):
    if train:
        xs,ys=mnist.train.next_batch(100)
        k=dropout
    else:
        xs,ys=mnist.test.images,mnist.test.labels
        k=1.0
    return {x:xs,y_:ys,keep_prob:k}

#train and test
saver=tf.train.Saver()
for i in range(max_steps):
    if i % 10 == 0:
        summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
        test_writer.add_summary(summary,i)
        print('Accuracy at step %s: %s' % (i,acc))
    else:
        summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True))
        train_writer.add_summary(summary,i) 
    
saver.save(sess,log_dir+"/model.ckpt",i)
    
train_writer.close()
test_writer.close()

    
            


        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
