#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:39:12 2018

This python file shows the basics of tensor flow usage

@author: suvosmac
"""

# import tensorflow
import tensorflow as tf

# constants in tensorflow would be stored in a tensor object
hello = tf.constant("Hello World!!")
hello # hello is a tensor object with data type as string

# Now we will create tensor flow sessions. THis is a class for running tensor flow operations
# A session object encapsulates an environment, in which operation objects are executed
sess = tf.Session() 

# Now we will run the tensor object created earlier under this session
sess.run(hello)

# Check the type
type(sess.run(hello))

# Operations - Multiple operations can be run under a tensor flow session
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("Operations with Constants")
    print('Addition :', sess.run(x + y))
    print('Subtraction :', sess.run(x - y))
    print('Multiplication :', sess.run(x * y))
    print('Division :', sess.run(x / y))
    
# Another object type in tensor flow is called placeholder which can accepts value. This is needed as sometimes
# we may not have a constant right away

x = tf.placeholder(tf.int64)
y = tf.placeholder(tf.int64)

add = tf.add(x,y)
sub = tf.subtract(x,y)
mult = tf.multiply(x,y)

with tf.Session() as sess:
    print("Operation with Placeholders")
    print('addition :',sess.run(add,feed_dict = {x:20, y:30}))
    print('subtraction :',sess.run(sub,feed_dict = {x:20, y:30}))
    print('Multiplication :',sess.run(mult,feed_dict = {x:20, y:30}))
    
# Now we will show matrix multiplication with tensorflow

import numpy as np

a = np.array([[5.0,5.0]]) # 1 X 2 matrix
b = np.array([[2.0],[2.0]]) # 2 X 1 matrix

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)