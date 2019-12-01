__author__ = 'abhishekyanamandra'
__ver__ = 0.1

import os
import time
import pickle
import librosa
import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

def ERmodel(INimg=[50,250,3], sr=16000, nH=[200,200,200], nO=8):
	h,w,c = INimg
	tf.reset_default_graph()
	seq_len = tf.placeholder(tf.int32, None) # For having Dynamic range RNN model
	signals = tf.placeholder(tf.float32, shape=[None, sr]) # For Audio Data
	frames = tf.placeholder(tf.float32, shape=[None, h, w, c]) # For Video data in banners
	Y = tf.placeholder(tf.float32, shape=[None, nO]) # For True label

	#Audio/Temporal part
	mfccs = tf_wav2mfcc(signals)[:,:,:,0] # Wav to MFCC conversion
	rnn_cell_ = [tf.keras.layers.GRUCell(nH_) for nH_ in nH]
	rnn_cells = tf.keras.layers.StackedRNNCells(rnn_cell_)
	outputs, states = tf.nn.dynamic_rnn(rnn_cells, mfccs, sequence_length=seq_len, dtype=tf.float32)
	out1A = tf.contrib.layers.fully_connected(outputs[:,-1,:], 200)

	#Video/Spatial part
	c1 = tf.contrib.layers.conv2d(frames, num_outputs=32, kernel_size=[3,3], padding="VALID")
	p1 = tf.contrib.layers.max_pool2d(c1, kernel_size=[2,2], stride=2)
	c2 = tf.contrib.layers.conv2d(p1, num_outputs=32, kernel_size=[5,5], padding="VALID")
	p2 = tf.contrib.layers.max_pool2d(c2, kernel_size=[2,2], stride=2)
	c3 = tf.contrib.layers.conv2d(p2, num_outputs=32, kernel_size=[5,5], padding="VALID")
	p3 = tf.contrib.layers.max_pool2d(c3, kernel_size=[2,2], stride=2)
	flt = tf.contrib.layers.flatten(p3)
	out1V = tf.contrib.layers.fully_connected(flt, 200)

	#Combining both the vectors of 200dims each
	combVec = out1A + out1V #tf.concat([out1A,out1V],axis=1) can also be tried
	out2 = tf.contrib.layers.fully_connected(combVec, 200)
	out = tf.contrib.layers.fully_connected(out2, 8, activation_fn=None)
	return signals, frames, Y, seq_len, out
