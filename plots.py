from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import Model
from array import array
import pandas as pd
import numpy as np
import keras
import sys


loss = pd.read_csv('./epochsLossFile.txt', sep="\n", header=None)
val_loss = pd.read_csv('./epochsValLossFile.txt', sep="\n", header=None)
print(loss)
print(val_loss)
epochs = [50, 100, 150, 200, 250, 300]
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, loss, 'ro')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.plot(epochs, val_loss, 'bo')
plt.title('Loss and Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Loss')
plt.savefig('epochsLoss.png')
plt.savefig('epochsLoss.png', bbox_inches='tight')
plt.show()
