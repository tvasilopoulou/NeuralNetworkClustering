from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten, Reshape
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.layers import Reshape
import matplotlib.pyplot as plt
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from array import array
import numpy as np
import keras
import sys

counter = 0
with open("./outputDataset.txt", "rb") as f:
	# f.seek(0, 2)
	# print(f.tell())
	
	magicNum = int.from_bytes(f.read(4), byteorder = "big")
	numOfImages = int.from_bytes(f.read(4), byteorder = "big")
	dx = int.from_bytes(f.read(4), byteorder = "big")
	dy = int.from_bytes(f.read(4), byteorder = "big")
	print(magicNum)
	print(numOfImages)
	print(dx)
	print(dy)
	dimensions = dx*dy
	buf = f.read(dimensions * numOfImages)
	images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	images = images.reshape(numOfImages, dx, dy)


trainLabels = []
with open("./train-labels.idx1-ubyte", "rb") as f:
	#read metadata from file
	magicNum = int.from_bytes(f.read(4), byteorder = "big")
	numOfImages = int.from_bytes(f.read(4), byteorder = "big")
	
	#read labels from file
	buf = f.read(numOfImages)
	trainLabels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
trainLabels = np.array(trainLabels)

plt.figure()

# Display the first image in training data
curr_img = np.reshape(images[0], (1,10))
curr_lbl = trainLabels[0]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(curr_lbl) + ")")	
plt.show()
