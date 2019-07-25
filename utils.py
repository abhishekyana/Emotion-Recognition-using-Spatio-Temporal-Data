__author__ = 'abhishekyanamandra'
__ver__ = 0.1
import os
import time
import pickle
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def tf_wav2mfcc(signals, sr=16000, frame_length=1024, frame_step=256, fft_length=1024):
	"""
	This function converts the wav file of sampleing rate sr into MFCCs(Mel-Frequency Cepstral Coefficients)
	It is highly optimised and can be moved over to GPU for parallelization.
	Args:
		signals: waveform
		sr: samping rate defaults is 16000Hz
		frame_length: each frame length default is 1024
		frame_step: each frame step to take in FT default is 256
		fft_length: FFT len default is 1024
	returns:
		log mel spectrograms
	"""
	stfts = tf.contrib.signal.stft(signals, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
	magnitude_spectrograms = tf.abs(stfts)
	num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
	lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000, 80
	linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz, upper_edge_hertz)
	mel_spectrograms = tf.tensordot( magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
	log_offset = 1e-6
	log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
	log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=3)
	return log_mel_spectrograms

def list2img(img, fps=5):
	"""
	image batch of fps size is converted into a single large banner where the temporal features in the video are preserved in a single image
	Args:
		img: Image is the batch of fps images
		fps: number images in the banner default is 5
	returns:
		banner of fps images aligned horizontally
	"""
	m,h,w,c = img.shape
	IMG=np.zeros((h, fps*w, c))
	for i in range(m):
		IMG[:, i*w:(i+1)*w, :] = img[i,:,:,:]/255 #normalizing the image
	return IMG

class dataset:
	def __init__(self, path='./DATA/DATA.d'):
		self.path = path
		self.data = pickle.load(open(path,'rb'))
		self.make_data()
		print("Done loading the dataset")
		self.m = self.AUDS.shape[0]
		self.make_batcher(ratio=0.8)
		
	def make_data(self, onehot=True):
		AUDS = []
		FACES = []
		LABELS = []
		for i,v in enumerate(self.data):
			a,f = self.data[v]
			t = a.size//16000
			ad=a[:t*16000]
			fd=f[:5*t]
			for ts in range(0,t):
				AUDS.append(ad[ts*16000:(ts+1)*16000])
				img = list2img(fd[ts*5:(ts+1)*5])
				FACES.append(img)
				lbval = int(v.split('-')[2])-1
				if onehot:
					lbv = np.zeros(8)
					lbv[lbval]=1.0
					lbval=lbv
				LABELS.append(lbval)
		self.AUDS = np.asarray(AUDS)
		self.FACES = np.asarray(FACES)
		self.LABELS = np.asarray(LABELS)
	
	def random_next_batch(self, bsize=32):
		inds = np.random.randnint(0,self.m,bsize)
		return self.AUDS[inds], self.FACES[inds], self.LABELS[inds]
	
	def make_batcher(self,ratio=0.6):
		self.bindex = {'train':0,
					   'val':0}
		self.shuffleinds = np.random.randint(0, self.m, self.m)
		self.rinds = {'train':self.shuffleinds[:int(ratio*self.m)],
					 'val':self.shuffleinds[int(ratio*self.m):]}
		self.mtr,self.mval = self.rinds['train'].shape[0], self.rinds['val'].shape[0]
	
	def next_batch(self,bsize=32,mode='train'):
		if (self.bindex[mode]+bsize)>=self.rinds[mode].shape[0]:
			self.bindex[mode]=0
		inds = np.arange(self.bindex[mode],self.bindex[mode]+bsize)
		inds = self.rinds[mode][inds]
		self.bindex[mode]+=bsize
		return self.AUDS[inds], self.FACES[inds], self.LABELS[inds]