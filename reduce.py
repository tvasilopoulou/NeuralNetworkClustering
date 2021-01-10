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
import struct
import keras
import sys

def encoder(inputImg, layers, maxFilters = 64, convFiltSize = (3, 3)):
	#set default parameters for predefined autoencoder model
	flag = 2
	#flag = amount of MaxPooling2D layers to be added
	numOfFilters = (int)(maxFilters/(pow(2, ((layers-6)/2) + 1)))
	#initial filters = calc how many sets of conv-batch-maxpooling are needed to reach the maxFilters
	countLayers = 0
	conv = inputImg
	while countLayers < layers:
		conv = Conv2D(numOfFilters, convFiltSize, activation='relu', padding='same')(conv)
		numOfFilters *= 2
		conv = BatchNormalization()(conv)
		countLayers += 2
		if flag != 0:
			#add MaxPooling on 2 first sets -> encoded/decoded image = maintain (28 x 28)
			conv = MaxPooling2D(pool_size=(2, 2))(conv)
			# conv = Dropout(0.1)(conv)
			# no dropout needed => Model does not overfit!
			countLayers += 1
			flag -= 1

	return conv
 
def decoder(conv, layers, maxFilters, convFiltSize, dx, dy):
	countLayers = 0
	while countLayers < layers :
		conv = Conv2DTranspose(maxFilters, convFiltSize, activation='relu', padding='same')(conv)
		maxFilters /= 2
		conv = BatchNormalization()(conv)
		countLayers += 2
		if countLayers >= layers - 4: #6
			conv = UpSampling2D((2, 2))(conv)
			# conv = Dropout(0.1)(conv)
			# no dropout needed => Model does not overfit!
			countLayers += 1
	decoded = Conv2DTranspose(1, convFiltSize, activation='sigmoid', padding='same')(conv) # 28 x 28 x 1
	# this is not considered a layer as it only fixes the output
	return decoded
 
 
def autoencoderBottleneck(dataset, queryset, layers, maxFilters, x, y, convFiltSize, batchSize, epochs, latentDim):
	images = []
	with open(dataset, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		images = images.reshape(numOfImages, dx, dy)
		f.close()
	# create shape of input for the model
	inChannel = 1
	inputImage = Input(shape = (dx, dy, inChannel))

	# create Model structure and compile, with loss metric being mean squared error and RMSprop optimizer
	encode = encoder(inputImage, int(layers)/2, maxFilters, convFiltSize)
	outShape = encode.shape
	flatLayer = Flatten()(encode)
	flatShape = flatLayer.shape 
	denseLayer = Dense(latentDim, activation = "relu")(flatLayer)

	print(flatShape)
	print(outShape)

	outputLayer = Dense(flatShape[1], activation = "softmax")(denseLayer)
	outputLayer = Reshape((outShape[1], outShape[2], outShape[3]))(outputLayer)


	autoencoder = Model(inputImage, decoder(outputLayer, int(layers)/2, maxFilters, convFiltSize, dx, dy))
	autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())


	xTrain, xValid, groundTrain, groundValid  = train_test_split(images, images, test_size=0.2, random_state=13)
	xTrain = np.array(xTrain).astype('float32') 
	groundTrain = np.array(groundTrain).astype('float32') 
	xValid = np.array(xValid).astype('float32') 
	groundValid = np.array(groundValid).astype('float32') 
	
	# print(np.concatenate((xTrain, xValid), axis=0).shape)

	# https://www.kite.com/python/answers/how-to-get-the-output-of-each-layer-of-a-keras-model-in-python
	autoencoder_train = autoencoder.fit(xTrain, groundTrain, \
                                        batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(xValid, groundValid))

	# autoencoder.save("./autoencoderModel.h5")

	with open(queryset, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		images = images.reshape(numOfImages, dx, dy)
		f.close()

	xQuery, xQValid, groundTrain, groundValid  = train_test_split(images, images, test_size=0.2, random_state=13)
	xQuery = np.array(xQuery).astype('float32') 
	groundTrain = np.array(groundTrain).astype('float32') 
	xQValid = np.array(xQValid).astype('float32') 
	groundValid = np.array(groundValid).astype('float32') 

	autoencoder_train = autoencoder.fit(xQuery, groundTrain, \
                                        batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(xQValid, groundValid))

	autoencoder.save("./autoencoderModel.h5")

	return [0, np.concatenate((xTrain, xValid), axis=0), np.concatenate((xQuery, xQValid), axis=0)]


def reduce(outputDataset, outputQueryset, latentDim, xTrain, xQuery):
	autoencoder = keras.models.load_model("./autoencoderModel.h5")


	latentVector = None
	for layer in autoencoder.layers:
		if layer.output_shape == (None, latentDim):
			# print(layer)
			latentVector = layer

	coordinates = (K.function([autoencoder.input], [latentVector.output])(xTrain)[0]).astype(np.ushort)
	qCoordinates = (K.function([autoencoder.input], [latentVector.output])(xQuery)[0]).astype(np.ushort)

	
	f = open(outputDataset, "ab")

	entry = 2051
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = len(coordinates)
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = 1
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = latentDim
	f.write(entry.to_bytes(4, byteorder='big'))
	for index in range(0, len(coordinates)):
		for k in range(0, len(coordinates[0])):
			entry = struct.pack('>H', coordinates[index][k])
			f.write(entry)

	f.close()

	f = open(outputQueryset, "ab")

	entry = 2051
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = len(qCoordinates)
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = 1
	f.write(entry.to_bytes(4, byteorder='big'))
	entry = latentDim
	f.write(entry.to_bytes(4, byteorder='big'))
	for index in range(0, len(qCoordinates)):
		for k in range(0, len(qCoordinates[0])):
			entry = struct.pack('>H', qCoordinates[index][k])
			f.write(entry)

	f.close()




if __name__ == "__main__":
	if(len(sys.argv) != 9):
		sys.exit("Please try running autoencoder again. Number of arguments was different than expected.\n");
	print("Welcome to Autoencoder. Before we get started, please provide us with a few parameter values. ")
	flag = 1
	# while user still chooses to run program
	while flag != 0:
		#provide hyperparameter values
		layers = input("Please enter number of layers: ") 
		maxFilters = input("Please enter number of filters (max): ") 
		# given MAX and not filters/layer
		x = input("Please enter a valid x dimension for the convolutional filters : ") 
		y = input("Please enter a valid y dimension for the convolutional filters : ") 
		convFiltSize = (int(x), int(y))
		batchSize = input("Please enter a batch size: ") 
		epochs = input("Please enter a number of epochs: ")
		latentQuery = input("Would you like a certain number of latent dimensions? (Y/n)")
		if latentQuery == 'Y' or latentQuery == 'y':
			latentDim = input("Please enter a number of latent dimensions: ") 
		else:
			latentDim = "10"
		if (not(layers.isdigit())) or (not(maxFilters.isdigit())) or (not(x.isdigit())) or (not(y.isdigit())) or (not(batchSize.isdigit())) or (not(epochs.isdigit())) or (not(latentDim.isdigit())):
			print("Something went wrong. Please try assigning integers as values.")
		else:
			promptStr = "Okay, so let's recap: you want " + str(layers) + " layers, "+ str(latentDim) + " latent dimensions, " + str(maxFilters) + " maximum number of filters, convolutional filter size" + str(convFiltSize) + ", " + str(batchSize) + " sized batches and " + str(epochs) + " epoch(s). Correct? (Y/n) "
			answer = input(promptStr)
			if answer == 'Y' or answer == 'y':
				layers = int(layers)
				maxFilters = int(maxFilters)
				batchSize = int(batchSize)
				epochs = int(epochs)
				latentDim = int(latentDim)
			else:
				# in case of error, repeat prompt
				print("Okay let's try again.")
				continue

		flag, xTrain, xQuery = autoencoderBottleneck(sys.argv[2], sys.argv[4], layers, maxFilters, x, y, convFiltSize, batchSize, epochs, latentDim)
		reduce(sys.argv[6], sys.argv[8], latentDim, xTrain, xQuery)