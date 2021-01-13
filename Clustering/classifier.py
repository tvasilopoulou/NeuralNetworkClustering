from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from autoencoder import encoder
import matplotlib.pyplot as plt
from keras.models import Model
from array import array
import numpy as np
import keras
import sys


def fullyConnected(enco, fcNodes):
	# flatten input provided from encoder
	flat = Flatten()(enco)
	# pass though user defined amount of fully connected nodes, linearly
	den = Dense(fcNodes, activation='relu')(flat)
	# model slightly overfitting => small dropout
	den = Dropout(0.2)(den)
	# utilize activation function => softmax yields best results
	out = Dense(10, activation='softmax')(den)
	return out


def classification(trainSet, trainLabelsSet, testSet, testLabelsSet, autoencoderPath, layers, fcNodes, batchSize, epochs):
	trainImages = []
	with open(trainSet, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		trainImages = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		trainImages = trainImages.reshape(numOfImages, dx, dy)

	# normalize array
	trainImages = np.array(trainImages).reshape(-1, dx, dy, 1).astype('float32') / 255.

	testImages = []
	with open(testSet, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		dx = int.from_bytes(f.read(4), byteorder = "big")
		dy = int.from_bytes(f.read(4), byteorder = "big")
		
		#read images from file
		dimensions = dx*dy
		buf = f.read(dimensions * numOfImages)
		testImages = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		testImages = testImages.reshape(numOfImages, dx, dy)

	# normalize array
	testImages = np.array(testImages).reshape(-1, dx, dy, 1).astype('float32') / 255.

	trainLabels = []
	with open(trainLabelsSet, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")
		
		#read labels from file
		buf = f.read(numOfImages)
		trainLabels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	trainLabels = np.array(trainLabels)

	testLabels = []
	with open(testLabelsSet, "rb") as f:
		#read metadata from file
		magicNum = int.from_bytes(f.read(4), byteorder = "big")
		numOfImages = int.from_bytes(f.read(4), byteorder = "big")

		#read labels from file
		buf = f.read(numOfImages)
		testLabels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

	testLabels = np.array(testLabels)
	
	# convert each label to binary vector this way: e.g. 2 -> [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
	trainCatLabels = to_categorical(trainLabels)
	testCatLabels = to_categorical(testLabels)

	# create shape of input for the model
	inChannel = 1
	inputImage = Input(shape = (dx, dy, inChannel))

	# split images into training and validation set
	xTrain, xValid, trainLabel, validLabel  = train_test_split(trainImages, trainCatLabels, test_size=0.2, random_state=13)

	# load autoencoder model provided by user
	autoencoder = keras.models.load_model(autoencoderPath)
	# if given encoder layers do not match autoencoder-encoder layers exit with error 
	if layers != (int(len(autoencoder.layers)/2) - 1):
		sys.exit("We are sorry to announce that no encoder exists with the given number of layers in your autoencoder. Please try again.")

	# create Model, utilize encoder/fully connected layers
	encode = encoder(inputImage, layers)
	fullModel = Model(inputImage,fullyConnected(encode, fcNodes))

	# copy encoder weights only into new model's encoder part
	for l1,l2 in zip(fullModel.layers[:layers],autoencoder.layers[0:layers]):
		l1.set_weights(l2.get_weights())

	# switch off training for encoder layers in new model -> train only fully connected layers
	for layer in fullModel.layers[0:(int(len(autoencoder.layers)/2) - 1)]:
		layer.trainable = False

	# utilize early stopping for loss/accuracy metrics
	keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')
	keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=2, verbose=0, mode='auto')
	keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
	keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='auto')
	
	# compile and train new model, different optimizer and loss metrics in this case and ground truth is now the label set
	fullModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	classifyTrain = fullModel.fit(xTrain, trainLabel, batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(xValid, validLabel))
	fullModel.save_weights('autoencoderClassification.h5')

	# turn encoder layers back on to fully train  model
	for layer in fullModel.layers[0:(int(len(autoencoder.layers)/2) - 1)]:
		layer.trainable = True

	# now compile and train for all layers
	fullModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
	classifyTrain = fullModel.fit(xTrain, trainLabel, batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(xValid, validLabel))
	fullModel.save_weights('classificationComplete.h5')

	# evalueate model and calculate amount of correct/incorrect predictions
	testEval = fullModel.evaluate(trainImages, trainCatLabels, verbose=0)
	predictedClasses = fullModel.predict(trainImages)
	predictedClasses = np.argmax(np.round(predictedClasses),axis=1)

	epochsLossFile = open("./outputTrainPredictions.txt", "a")
	for i in range(len(predictedClasses)):
		epochsLossFile.write(str(predictedClasses[i]) + "\n")
	epochsLossFile.close()

	# print(predictedClasses)

	# comments below are to create the files used for graphs

	# epochsLossFile = open("/content/drive/My Drive/Colab Notebooks/epochsLossFile.txt", "a")
	# epochsLossFile.write(str(autoencoder_train.history['loss'][-1]) + "\n")
	# epochsLossFile.close()

	# epochsValLossFile = open("/content/drive/My Drive/Colab Notebooks/epochsValLossFile.txt", "a")
	# epochsValLossFile.write(str(autoencoder_train.history['val_loss'][-1]) + "\n")
	# epochsValLossFile.close() 

	# epochsAccuracyFile = open("/content/drive/My Drive/Colab Notebooks/epochsAccuracyFile.txt", "a")
	# epochsAccuracyFile.write(str(autoencoder_train.history['accuracy'][-1]) + "\n")
	# epochsAccuracyFile.close()

	# epochsValAccuracyFile = open("/content/drive/My Drive/Colab Notebooks/epochsValAccuracyFile.txt", "a")
	# epochsValAccuracyFile.write(str(autoencoder_train.history['val_accuracy'][-1]) + "\n")
	# epochsValAccuracyFile.close() 

	inp = input("We have produced the classification model. Would you like to start with predictions? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		print("Started making predictions...")

		correct = np.where(predictedClasses==testLabels)[0]
		print ("Found " + str(len(correct))+ " correct labels" )
		# if user is interested, show data about correctness
		for i, correct in enumerate(correct[:12]):
			plt.subplot(4,3,i+1)
			plt.imshow(testImages[correct].reshape(28,28), cmap='gray', interpolation='none')
			plt.title("Predicted {}, Class {}".format(predictedClasses[correct], testLabels[correct]))
			plt.tight_layout()
		plt.savefig('correctPredictions.png')
		plt.savefig('correctPredictions.png', bbox_inches='tight')
		plt.show()

		incorrect = np.where(predictedClasses!=testLabels)[0]
		print ("Found " + str(len(incorrect))+ " incorrect labels" )
		plt.figure()

		for i, incorrect in enumerate(incorrect[:12]):
			plt.subplot(4,3,i+1)
			plt.imshow(testImages[incorrect].reshape(28,28), cmap='gray', interpolation='none')
			plt.title("Predicted {}, Class {}".format(predictedClasses[incorrect], testLabels[incorrect]))
			plt.tight_layout()
		plt.savefig('incorrectPredictions.png')
		plt.savefig('incorrectPredictions.png', bbox_inches='tight')
		plt.show()

		print("More specifically:\n")
		# accuracy, loss, precision, recall, f score appear now
		print(classification_report(testLabels, predictedClasses))

	inp = input("Would you like to plot your experiment's loss and accuracy results? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		# again, plot for current run
		accuracy = classifyTrain.history['accuracy']
		val_accuracy = classifyTrain.history['val_accuracy']
		loss = classifyTrain.history['loss']
		val_loss = classifyTrain.history['val_loss']
		epochs = range(len(accuracy))
		plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
		plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
		plt.title('Training and validation accuracy')
		plt.legend()
		plt.figure()
		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, 'b', label='Validation loss')
		plt.title('Training and validation loss')
		plt.legend()
		plt.savefig('overfittingClassCheck.png')
		plt.savefig('overfittingClassCheck.png', bbox_inches='tight')
		plt.show()


	inp = input("Would you like to repeat your experiment with different hyperparameter values? (Y/n) ")
	if inp == 'Y' or inp == 'y':
		return 1
		
	return 0



if __name__ == "__main__":

	# python classifier.py -d .\train-images.idx3-ubyte -dl .\train-labels.idx1-ubyte -t .\t10k-images.idx3-ubyte -tl .\t10k-labels.idx1-ubyte -model .\autoencoder.h5

	if(len(sys.argv) != 11):
		sys.exit("Please try running classifier again. Number of arguments was different than expected.");
	
	# argumements should be given in specific order
	if (sys.argv[1]!= "-d" and sys.argv[3]!= "-dl" and sys.argv[5]!= "-t" and sys.argv[7]!= "-tl" and sys.argv[9]!= "-model"):
		sys.exit("Please try running classifier again. Arguments were either incorrect or in an incorrect order.");


	print("Welcome to Classifier. Before we get started, please provide us with a few parameter values. ")
	flag = 1
	while flag != 0:
		layers = input("Please enter number of encoder layers: ") 
		fcNodes = input("Please enter number of fully connected nodes: ") 
		batchSize = input("Please enter a batch size: ") 
		epochs = input("Please enter a number of epochs: ") 
		if (not(layers.isdigit())) or (not(fcNodes.isdigit())) or (not(batchSize.isdigit())) or (not(epochs.isdigit())):
			print("Something went wrong. Please try assigning integers as values.")
		else:
			promptStr = "Okay, so let's recap: you want " + str(layers) + " layers, " + str(fcNodes) + " fully connected nodes, " + str(batchSize) + " sized batches and " + str(epochs) + " epoch(s). Correct? (Y/n) "
			answer = input(promptStr)
			if answer == 'Y' or answer == 'y':
				layers = int(layers)
				fcNodes = int(fcNodes)
				batchSize = int(batchSize)
				epochs = int(epochs)
			else:
				print("Okay let's try again.")
				continue

		flag = classification(sys.argv[2], sys.argv[4], sys.argv[6], sys.argv[8], sys.argv[10] , layers, fcNodes, batchSize, epochs)