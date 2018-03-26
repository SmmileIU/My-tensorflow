#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:50:15 2018

@author: zhao
"""

import tensorflow as tf
import numpy as np
import input_data
mnist=input_data.read_data_sets("/tmp/MNIST_data/",one_hot=True)

#dingyi train canshu
learning_rate=0.01
max_samples=400000
batch_size=128
display_step=10

# Network Parameters
n_input=28
n_steps=28
n_hidden=256
n_classes=10

#dingyi bianlian
x=tf.placeholder("float",[None,n_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

weights=tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
biases=tf.Variable(tf.random_normal([n_classes]))

#the function to creat Bi-LSTM wangluo
def BiRNN(x,weights,biases):
    
    x=tf.transpose(x,[1,0,2])
    x=tf.reshape(x,[-1,n_input])
    x=tf.split(x,n_steps)
    
    lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    
    outputs,_,_=tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights)+biases

#jisuan loss and optimizer
pred=BiRNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()


#train and test
with tf.Session() as sess:
    sess.run(init)
    step=1
    #train
    while step * batch_size < max_samples:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step == 0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter"+str(step*batch_size)+",Minibatch Loss="+"{:.6f}".format(loss)+\
                  ",Training Accuracy="+"{:.5f}".format(acc))
        step += 1       
    print("Optimization Finished!")
    
    #Test
    test_len=10000
    test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:test_len]
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

