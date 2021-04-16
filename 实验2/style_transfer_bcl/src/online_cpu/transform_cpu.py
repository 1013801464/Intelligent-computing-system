#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import argparse
import numpy as np
import cv2 as cv
import time
from power_diff_numpy import *

os.putenv('MLU_VISIBLE_DEVICES','')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('ori_pb')
    parser.add_argument('ori_power_diff_pb')
    parser.add_argument('numpy_pb')
    args = parser.parse_args()
    return args

def run_ori_pb():  # original PB
    args = parse_arg()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(args.ori_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]

    g = tf.compat.v1.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.ori_pb,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))
        with tf.compat.v1.Session(config=config) as sess:
            sess.graph.as_default()
            sess.run(tf.compat.v1.global_variables_initializer())

            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')

            start_time = time.time()
            ret = sess.run(output_tensor, feed_dict={input_tensor:[X]})
            end_time = time.time()
            print("C++ inference(CPU) origin pb time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)


def run_ori_power_diff_pb():  # original power_diff pb
    args = parse_arg()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(args.ori_power_diff_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]

    g = tf.compat.v1.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.ori_power_diff_pb,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))
        with tf.compat.v1.Session(config=config) as sess:
            # YKH：完成PowerDifference Pb模型的推理
            sess.graph.as_default()
            sess.run(tf.compat.v1.global_variables_initializer())

            input_tensor = sess.graph.get_tensor_by_name('X_content:0')  #　cmt: correct the tensor name
            pow_tensor = sess.graph.get_tensor_by_name('moments_15/PowerDifference_z:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')

            start_time = time.time()
            ret = sess.run(output_tensor, feed_dict={input_tensor:[X], pow_tensor:2.0})
            end_time = time.time()
            print("C++ inference(CPU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)

def run_numpy_pb():  # power diff pb
    args = parse_arg()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    model_name = os.path.basename(args.numpy_pb).split(".")[0]
    image_name = os.path.basename(args.image).split(".")[0]

    g = tf.compat.v1.Graph()
    with g.as_default():
        with tf.gfile.FastGFile(args.numpy_pb,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        img = cv.imread(args.image)
        X = cv.resize(img, (256, 256))
        with tf.compat.v1.Session(config=config) as sess:
            # YKH：完成Numpy版本 Pb模型的推理
            sess.graph.as_default()
            sess.run(tf.compat.v1.global_variables_initializer())

            start_time = time.time()
            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            conv2d_13 = sess.graph.get_tensor_by_name('Conv2D_13:0')
            stop_gradient = sess.graph.get_tensor_by_name('moments_15/StopGradient:0')
            [ret1, ret2] = sess.run([conv2d_13, stop_gradient], feed_dict={input_tensor:[X]})
            
            pd_result_tensor = sess.graph.get_tensor_by_name('moments_15/PowerDifference:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')
            ret = sess.run(output_tensor, feed_dict={input_tensor:[X], pd_result_tensor:(ret1-ret2)**2})
            end_time = time.time()
            print("Numpy inference(CPU) time is: ",end_time-start_time)
            img1 = tf.reshape(ret,[256,256,3])
            img_numpy = img1.eval(session=sess)
            cv.imwrite(image_name + '_' + model_name + '_cpu.jpg',img_numpy)


if __name__ == '__main__':
    run_ori_pb()
    run_ori_power_diff_pb()
    run_numpy_pb()
