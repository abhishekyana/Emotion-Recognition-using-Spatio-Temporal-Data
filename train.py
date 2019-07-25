__author__ = 'abhishekyanamandra'
__ver__ = 0.1

import os
import time
import pickle
import librosa
import numpy as np
from utils import *
from model import ERmodel
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__=="__main__":
	D = dataset()
	print("Data dimensions are: ",D.AUDS.shape, D.FACES.shape, D.LABELS.shape)
	signals, frames, Y, seq_len, out = ERmodel(INimg=[50,250,3], sr=16000, nH=[200,200,200], nO=8)
	
	#Loss and accuracy
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,out))
	acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, axis=1), tf.argmax(Y, axis=1)), tf.float32))
	
	# Optimizer 
	opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) # Learning rate is fixed, but can be changed

	# Actual Training
	stime = time.time()
	epochs = 100 # Add as args parser
	evali = 2 #evaluate interval
	bsize = 64 # Batch size
	Ts = 16000//256 - 3 # Sampled rate to MFCCs timesteps conversion

	L, Lval=[], []
	ACC, ACCval=[], []
	preds, predsval = [], []
	tottr, totval = 1 + D.mtr//bsize, 1 + D.mval//bsize # Total training and Validation steps to run the counter for Dataloader
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		#Training
		for ep in range(epochs+1):
			for batch in range(0, D.mtr, bsize):
				xab,xvb,yb = D.next_batch(bsize)
				fd = {signals:xab,frames:xvb, Y:yb, seq_len:Ts}
				if ep==0:
					ls, ac = sess.run([loss,acc], feed_dict=fd)
				else:
					ls, ac, _ = sess.run([loss,acc,opt], feed_dict=fd)
					if ep==epochs:
						preds.append(sess.run([out,Y],feed_dict=fd))
				L.append(ls)
				ACC.append(ac)
			if ep%evali==0:
				print(f"After epoch {ep:3.0f} Training;   Accuracy is {100*sum(ACC[-tottr:])/tottr:3.3f}% and loss is {sum(L[-tottr:])/tottr:3.3f}")
				for batch in range(0, D.mval, bsize):
					xabval, xvbval, ybval = D.next_batch(bsize, 'val')
					fdval = {signals:xabval, frames:xvbval, Y:ybval, seq_len:Ts}#,
					lsval, acval = sess.run([loss, acc], feed_dict=fdval)
					Lval.append(lsval)
					ACCval.append(acval)
					if ep==epochs:
						predsval.append(sess.run([out,Y], feed_dict=fdval))
				print(f"After epoch {ep:3.0f} Validation; Accuracy is {100*sum(ACCval[-totval:])/totval:3.3f}% and loss is {sum(Lval[-totval:])/totval:3.3f}")
				print()
		print(f"Done in {time.time()-stime:5.5f} Secs")
		saver = tf.train.Saver()
		svpath = './trained/ERtrained/'
		os.makedirs(svpath)
		print(f"Saved at {saver.save(sess, svpath+'Model')}")
	print(f"Done in {time.time()-stime:5.5f} Secs")