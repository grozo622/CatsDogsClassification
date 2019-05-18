
# coding: utf-8

# In[7]:


#part 2 of Assignment #2, continuing experiments.

# Common imports for our work
import os 
import numpy as np
import tensorflow as tf
from datetime import datetime  # for time-stamps on file names

import time

from matplotlib import pyplot as plt  # for display of images

RANDOM_SEED = 9999

from keras import layers
from keras import models
from keras import optimizers


# In[2]:


#CatsDogs# 

#ONLY using 64x64 black & white images (1 channel) in this study.

# Documentation on npy binary format for saving numpy arrays for later use
#     https://towardsdatascience.com/
#             why-you-should-start-using-npy-file-more-often-df2a13cc0161
# Under the working directory, data files are in directory cats_dogs_64_128 
# Read in cats and dogs grayscale 64x64 files to create training data
cats_1000_64_64_1 = np.load('cats_1000_64_64_1.npy')
dogs_1000_64_64_1 = np.load('dogs_1000_64_64_1.npy')


# In[3]:


X_cat_dog = np.concatenate((cats_1000_64_64_1, dogs_1000_64_64_1), axis = 0) 

#X_cat_dog.shape   #so we combined all the cats/dogs together for 2000 total.
#created 4096 just by 64*64

X_cat_dog = X_cat_dog.reshape((2000, 64, 64, 1))
X_cat_dog = X_cat_dog.astype('float32')/255

X_cat_dog.shape


# In[4]:


# Define the labels to be used 1000 cats = 0 1000 dogs = 1
y_cat_dog = np.concatenate((np.zeros((1000), dtype = np.int32), 
                      np.ones((1000), dtype = np.int32)), axis = 0)

#so zero = cats, one = dogs.

y_cat_dog.shape


# In[5]:


# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test =     train_test_split(X_cat_dog, y_cat_dog, test_size=0.20, 
                     random_state = RANDOM_SEED)

#use the random_seed created from before to randomly split the data...


# In[6]:


X_train = X_train[400:]
X_val = X_train[:400]

y_train = y_train[400:]
y_val = y_train[:400]

y_val.shape


# In[37]:


#add in a Conv2D (64) and maxPooling2D ((2, 2))

#now, use dropout (0.05, 0.15, 0.25, 0.50)

#change dense 256 -> 512 ...

#only use low batch_size=64

model30 = models.Sequential()
model30.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model30.add(layers.MaxPooling2D((2, 2)))
model30.add(layers.Flatten())
model30.add(layers.Dropout(0.05))
model30.add(layers.Dense(256, activation='relu'))
model30.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model30.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model30.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[38]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('1layer_0.05dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('1layer_0.05dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[39]:


model31 = models.Sequential()
model31.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model31.add(layers.MaxPooling2D((2, 2)))
model31.add(layers.Flatten())
model31.add(layers.Dropout(0.15))
model31.add(layers.Dense(256, activation='relu'))
model31.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model31.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model31.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[40]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('1layer_0.15dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('1layer_0.15dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[41]:


model32 = models.Sequential()
model32.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model32.add(layers.MaxPooling2D((2, 2)))
model32.add(layers.Flatten())
model32.add(layers.Dropout(0.25))
model32.add(layers.Dense(256, activation='relu'))
model32.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model32.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model32.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[42]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('1layer_0.25dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('1layer_0.25dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[43]:


model33 = models.Sequential()
model33.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model33.add(layers.MaxPooling2D((2, 2)))
model33.add(layers.Flatten())
model33.add(layers.Dropout(0.50))
model33.add(layers.Dense(256, activation='relu'))
model33.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model33.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model33.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[44]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('1layer_0.50dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('1layer_0.50dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[45]:


#fc/dense layer 512... everything else the same dropout


# In[64]:


model34 = models.Sequential()
model34.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model34.add(layers.MaxPooling2D((2, 2)))
model34.add(layers.Flatten())
model34.add(layers.Dropout(0.05))
model34.add(layers.Dense(512, activation='relu'))
model34.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model34.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model34.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[47]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('1layer_0.05dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('1layer_0.05dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[48]:


model35 = models.Sequential()
model35.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model35.add(layers.MaxPooling2D((2, 2)))
model35.add(layers.Flatten())
model35.add(layers.Dropout(0.15))
model35.add(layers.Dense(512, activation='relu'))
model35.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model35.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model35.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[49]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('1layer_0.15dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('1layer_0.15dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[50]:


model36 = models.Sequential()
model36.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model36.add(layers.MaxPooling2D((2, 2)))
model36.add(layers.Flatten())
model36.add(layers.Dropout(0.25))
model36.add(layers.Dense(512, activation='relu'))
model36.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model36.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model36.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[51]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('1layer_0.25dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('1layer_0.25dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[52]:


model37 = models.Sequential()
model37.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model37.add(layers.MaxPooling2D((2, 2)))
model37.add(layers.Flatten())
model37.add(layers.Dropout(0.50))
model37.add(layers.Dense(512, activation='relu'))
model37.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model37.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model37.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[53]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('1layer_0.50dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('1layer_0.50dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[54]:


test_loss30, test_acc30 = model30.evaluate(X_test, y_test)
test_loss31, test_acc31 = model31.evaluate(X_test, y_test)
test_loss32, test_acc32 = model32.evaluate(X_test, y_test)
test_loss33, test_acc33 = model33.evaluate(X_test, y_test)
test_loss34, test_acc34 = model34.evaluate(X_test, y_test)
test_loss35, test_acc35 = model35.evaluate(X_test, y_test)
test_loss36, test_acc36 = model36.evaluate(X_test, y_test)
test_loss37, test_acc37 = model37.evaluate(X_test, y_test)


# In[63]:


#test_acc30 #0.6125
#test_acc31 #0.6525
#test_acc32 #0.58
#test_acc33 #0.635
#test_acc34 #0.6075
#test_acc35 #0.6575
#test_acc36 #0.6475
#test_acc37 #0.6625


# In[ ]:


#so dropout seems to help a lot, 0.25 had best results.

#512 nodes and dropout ~0.25... but maybe check 0.20, 0.30, 0.35...
#it is sensitive to number of nodes, type of problem...

#page 110 of Chollet also shows could do dropout after every layer


# In[ ]:


#add one more conv2D layer and MaxPooling layer, see how much improves...


# In[65]:


model40 = models.Sequential()
model40.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model40.add(layers.MaxPooling2D((2, 2)))
model40.add(layers.Conv2D(128, (3, 3), activation='relu'))
model40.add(layers.MaxPooling2D((2, 2)))
model40.add(layers.Flatten())
model40.add(layers.Dropout(0.05))
model40.add(layers.Dense(256, activation='relu'))
model40.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model40.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model40.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[66]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 256 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('2layers_0.05dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 256 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('2layers_0.05dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[67]:


model41 = models.Sequential()
model41.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model41.add(layers.MaxPooling2D((2, 2)))
model41.add(layers.Conv2D(128, (3, 3), activation='relu'))
model41.add(layers.MaxPooling2D((2, 2)))
model41.add(layers.Flatten())
model41.add(layers.Dropout(0.15))
model41.add(layers.Dense(256, activation='relu'))
model41.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model41.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model41.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[68]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 256 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('2layers_0.15dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 256 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('2layers_0.15dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[13]:


model42 = models.Sequential()
model42.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model42.add(layers.MaxPooling2D((2, 2)))
model42.add(layers.Conv2D(128, (3, 3), activation='relu'))
model42.add(layers.MaxPooling2D((2, 2)))
model42.add(layers.Flatten())
model42.add(layers.Dropout(0.25))
model42.add(layers.Dense(256, activation='relu'))
model42.add(layers.Dense(1, activation='sigmoid'))

##########################################################


model42.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model42.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[15]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 256 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('2layers_0.25dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 256 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('2layers_0.25dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[16]:


model43 = models.Sequential()
model43.add(layers.Conv2D(64, (3, 3), activation='relu',
                      input_shape=(64, 64, 1)))
model43.add(layers.MaxPooling2D((2, 2)))
model43.add(layers.Conv2D(128, (3, 3), activation='relu'))
model43.add(layers.MaxPooling2D((2, 2)))
model43.add(layers.Flatten())
model43.add(layers.Dropout(0.50))
model43.add(layers.Dense(256, activation='relu'))
model43.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model43.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model43.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[17]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 256 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('2layers_0.50dropout_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 256 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('2layers_0.50dropout_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[18]:


#512 nodes

model44 = models.Sequential()
model44.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model44.add(layers.MaxPooling2D((2, 2)))
model44.add(layers.Conv2D(128, (3, 3), activation='relu'))
model44.add(layers.MaxPooling2D((2, 2)))
model44.add(layers.Flatten())
model44.add(layers.Dropout(0.05))
model44.add(layers.Dense(512, activation='relu'))
model44.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model44.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model44.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[19]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 512 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('2layers_0.05dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 512 Nds, 0.05 d.o.)')
plt.legend()

plt.savefig('2layers_0.05dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[20]:


model45 = models.Sequential()
model45.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model45.add(layers.MaxPooling2D((2, 2)))
model45.add(layers.Conv2D(128, (3, 3), activation='relu'))
model45.add(layers.MaxPooling2D((2, 2)))
model45.add(layers.Flatten())
model45.add(layers.Dropout(0.15))
model45.add(layers.Dense(512, activation='relu'))
model45.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model45.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model45.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[21]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 512 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('2layers_0.15dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 512 Nds, 0.15 d.o.)')
plt.legend()

plt.savefig('2layers_0.15dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[22]:


model46 = models.Sequential()
model46.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model46.add(layers.MaxPooling2D((2, 2)))
model46.add(layers.Conv2D(128, (3, 3), activation='relu'))
model46.add(layers.MaxPooling2D((2, 2)))
model46.add(layers.Flatten())
model46.add(layers.Dropout(0.25))
model46.add(layers.Dense(512, activation='relu'))
model46.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model46.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model46.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[23]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 512 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('2layers_0.25dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 512 Nds, 0.25 d.o.)')
plt.legend()

plt.savefig('2layers_0.25dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[24]:


model47 = models.Sequential()
model47.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model47.add(layers.MaxPooling2D((2, 2)))
model47.add(layers.Conv2D(128, (3, 3), activation='relu'))
model47.add(layers.MaxPooling2D((2, 2)))
model47.add(layers.Flatten())
model47.add(layers.Dropout(0.50))
model47.add(layers.Dense(512, activation='relu'))
model47.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model47.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model47.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[25]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (2 Conv/MaxPool lay, 512 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('2layers_0.50dropout_512_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (2 Conv/MaxPool lay, 512 Nds, 0.50 d.o.)')
plt.legend()

plt.savefig('2layers_0.50dropout_512_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[26]:


test_loss40, test_acc40 = model40.evaluate(X_test, y_test)
test_loss41, test_acc41 = model41.evaluate(X_test, y_test)
test_loss42, test_acc42 = model42.evaluate(X_test, y_test)
test_loss43, test_acc43 = model43.evaluate(X_test, y_test)
test_loss44, test_acc44 = model44.evaluate(X_test, y_test)
test_loss45, test_acc45 = model45.evaluate(X_test, y_test)
test_loss46, test_acc46 = model46.evaluate(X_test, y_test)
test_loss47, test_acc47 = model47.evaluate(X_test, y_test)


# In[35]:


#test_acc40 #0.645
#test_acc41 #0.6275
#test_acc42 #0.64
#test_acc43 #0.6275
#test_acc44 #0.6725
#test_acc45 #0.66
#test_acc46 #0.665
#test_acc47 #0.66

