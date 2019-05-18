
# coding: utf-8

# In[2]:


#Geoffrey Rozo
#MSDS 458
#Assignment #2

# coding: utf-8

# Initial deep neural network set-up from 
# Géron, A. 2017. Hands-On Machine Learning with Scikit-Learn 
#    & TensorFlow: Concepts, Tools, and Techniques to Build 
#    Intelligent Systems. Sebastopol, Calif.: O'Reilly. 
#    [ISBN-13 978-1-491-96229-9] 
#    Source code available at https://github.com/ageron/handson-ml
#    See file 10_introduction_to_artificial_neural_networks.ipynb 
#    Revised from MNIST to Cats and Dogs to begin Assignment 7
#    #CatsDogs# comment lines show additions/revisions for Cats and Dogs

#Also, Chollet, F. (2018). Deep learning with Python.
#Shelter Island, NY: Manning Publications.
#Gave helpful ideas/used some code in chapter 5 for CNN layers, regulariz.,
#dropout, and further research...

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports for our work
import os 
import numpy as np
import tensorflow as tf
from datetime import datetime  # for time-stamps on file names

import time

RANDOM_SEED = 9999


# In[3]:


#CatsDogs# 

#ONLY using 64x64 pixel images (black/white 1 channel) for this assignment.

# Documentation on npy binary format for saving numpy arrays for later use
#     https://towardsdatascience.com/
#             why-you-should-start-using-npy-file-more-often-df2a13cc0161
# Under the working directory, data files are in directory cats_dogs_64_128 
# Read in cats and dogs grayscale 64x64 files to create training data
cats_1000_64_64_1 = np.load('cats_1000_64_64_1.npy')
dogs_1000_64_64_1 = np.load('dogs_1000_64_64_1.npy')


# In[4]:


cats_1000_64_64_1.shape


# In[5]:


dogs_1000_64_64_1.shape


# In[6]:


from matplotlib import pyplot as plt  # for display of images
def show_grayscale_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


# In[7]:


# Examine first cat and first dog grayscale images
show_grayscale_image(cats_1000_64_64_1[0,:,:,0])

plt.savefig('cat_example_64_1.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


# In[8]:


show_grayscale_image(dogs_1000_64_64_1[0,:,:,0])

plt.savefig('dog_example_64_1.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


# In[9]:


# Work the data for cats and dogs numpy arrays 
# These numpy arrays were generated in previous data prep work
# Stack the numpy arrays for the inputs
X_cat_dog = np.concatenate((cats_1000_64_64_1, dogs_1000_64_64_1), axis = 0) 
#X_cat_dog = X_cat_dog.reshape(-1,width*height) # note coversion to 4096 inputs


# In[10]:


#X_cat_dog.shape   #so we combined all the cats/dogs together for 2000 total.
#created 4096 just by 64*64

X_cat_dog = X_cat_dog.reshape((2000, 64, 64, 1))
X_cat_dog = X_cat_dog.astype('float32')/255

X_cat_dog.shape


# In[11]:


# Scikit Learn for min-max scaling of the data
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(np.array([0., 255.]).reshape(-1,1))

#scaling to fit the features possible values from 0 - 255

#X_cat_dog_min_max = scaler.transform(X_cat_dog)


# In[12]:


# Define the labels to be used 1000 cats = 0 1000 dogs = 1
y_cat_dog = np.concatenate((np.zeros((1000), dtype = np.int32), 
                      np.ones((1000), dtype = np.int32)), axis = 0)

#so zero = cats, one = dogs.

y_cat_dog.shape


# In[13]:


# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test =     train_test_split(X_cat_dog, y_cat_dog, test_size=0.20, 
                     random_state = RANDOM_SEED)

#use the random_seed created from before to randomly split the data...


# In[14]:


X_train = X_train[400:]
X_val = X_train[:400]

y_train = y_train[400:]
y_val = y_train[:400]

y_val.shape


# In[15]:


from keras.utils import to_categorical

train_labels = to_categorical(y_train)
val_labels = to_categorical(y_val)
test_labels = to_categorical(y_test)


# In[16]:


#if I want to, can take different sizes of training set to see how
#different numbers of data inputs are impacted by a more complex model
# (with more hidden layers/many hyperparameters)


# In[19]:


from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[20]:


model.summary()


# In[21]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[22]:


processing_time = []
    
start_time = time.clock()

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[21]:


history_dict = history.history
history_dict.keys()


# In[22]:


#Cholet - Deep Learning with Python (pages 137-138) for plots:

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()


# In[23]:


#could even try using page 139 (ImageDataGenerator) to augment the data


# In[24]:


#experimenting: make the above better with maxpooling, more layers, dropout.

from keras import layers
from keras import models
from keras import optimizers

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))


# In[25]:


model2.summary()


# In[26]:


model2.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[27]:


processing_time = []
    
start_time = time.clock()

history = model2.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[28]:


#Visually, this should look similar to page 142 in Cholet...
#more epochs shows the increasing trend/decreasing loss


# In[29]:


#Cholet - Deep Learning with Python (pages 137-138) for plots:

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()


# In[30]:


#cool... now play with this for Assignment 2... adjust number of convnets
#try dropout...

#page 142-143 recommend using a pretrained convnet...
#could try the VGG16 on Assignment 4?


# In[31]:


#Change batch size, try batch size of 64 ... 128 ... 256 ... 512 ...


# In[32]:


#Experiment 3

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dense(256, activation='relu'))
model3.add(layers.Dense(1, activation='sigmoid'))


# In[33]:


model3.summary()


# In[34]:


model3.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[35]:


processing_time = []
    
start_time = time.clock()

history = model3.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=256,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[36]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()


# In[37]:


#so, increasing batch_size from 128 -> 256 resulted in worse performance.

#next, try batch_size 64... then get into dropout.


# In[38]:


model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(64, (3, 3), activation='relu'))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Flatten())
model4.add(layers.Dense(256, activation='relu'))
model4.add(layers.Dense(1, activation='sigmoid'))


# In[39]:


model4.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[40]:


processing_time = []
    
start_time = time.clock()

history = model4.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[41]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()


# In[42]:


#so, smaller batch_size results in better results, for this problem.
#so far, batch_size = 64 best.


# In[43]:


model5 = models.Sequential()
model5.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Flatten())
model5.add(layers.Dense(512, activation='relu'))
model5.add(layers.Dense(1, activation='sigmoid'))


# In[44]:


model5.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[45]:


model5.summary()


# In[46]:


processing_time = []
    
start_time = time.clock()

history = model5.fit(X_train,
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
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()


# In[ ]:


#from above, model 3 and model 4 are the same except for batch_size being 256 in model 3 and then 64 in model 4.
#the lower batch size showed better accuracy/loss results. Lower batch size took about 40 seconds longer but was
#worth the performance boost.


# In[ ]:


#Assignment #2: Test on different ==> batch_size = 64, 128, 256
#                                 ==> 1 hidden, 1 Conv2D/maxpool, 2 Conv2D/maxpool, 3Conv2D/maxpool
#                                 ==> layers.dense(256 and 512)
#                                 ==> dropout(0.05, 0.15, 0.25, 0.50)
#           all activations       ==> relu except for last sigmoid layer
#           all epochs            ==> 50


# In[ ]:


#Start with one layer (256, then 512 nodes) with an output sigmoid layer.

#then we will follow the Chollet pages 130-142 on adding Conv2D layers with 32, 64, 128 nodes, and even dropout.


# In[15]:


#1 layer, 256 nodes

from keras import layers
from keras import models
from keras import optimizers


# In[16]:


model10 = models.Sequential()
model10.add(layers.Flatten())
model10.add(layers.Dense(256, activation='relu'))
model10.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model10.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model10.fit(X_train,
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
plt.title('Train & Valid Acc. (1 lay, 256 Nds, 64 btch)')
plt.legend()

plt.savefig('1layer_256_64_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 256 Nds, 64 btch)')
plt.legend()

plt.savefig('1layer_256_64_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[18]:


model11 = models.Sequential()
model11.add(layers.Flatten())
model11.add(layers.Dense(256, activation='relu'))
model11.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model11.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model11.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128,
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
plt.title('Train & Valid Acc. (1 lay, 256 Nds, 128 btch)')
plt.legend()

plt.savefig('1layer_256_128_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 256 Nds, 128 btch)')
plt.legend()

plt.savefig('1layer_256_128_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[20]:


model12 = models.Sequential()
model12.add(layers.Flatten())
model12.add(layers.Dense(256, activation='relu'))
model12.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model12.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model12.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=256,
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
plt.title('Train & Valid Acc. (1 lay, 256 Nds, 256 btch)')
plt.legend()

plt.savefig('1layer_256_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 256 Nds, 256 btch)')
plt.legend()

plt.savefig('1layer_256_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[22]:


#1 layer, 512 nodes


# In[23]:


model13 = models.Sequential()
model13.add(layers.Flatten())
model13.add(layers.Dense(512, activation='relu'))
model13.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model13.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model13.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[24]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 lay, 512 Nds, 64 btch)')
plt.legend()

plt.savefig('1layer_512_64_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 512 Nds, 64 btch)')
plt.legend()

plt.savefig('1layer_512_64_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[25]:


model14 = models.Sequential()
model14.add(layers.Flatten())
model14.add(layers.Dense(512, activation='relu'))
model14.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model14.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model14.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[26]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 lay, 512 Nds, 128 btch)')
plt.legend()

plt.savefig('1layer_512_128_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 512 Nds, 128 btch)')
plt.legend()

plt.savefig('1layer_512_128_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[27]:


model15 = models.Sequential()
model15.add(layers.Flatten())
model15.add(layers.Dense(512, activation='relu'))
model15.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model15.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model15.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=256,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[28]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 lay, 512 Nds, 256 btch)')
plt.legend()

plt.savefig('1layer_512_256_Acc.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 lay, 512 Nds, 256 btch)')
plt.legend()

plt.savefig('1layer_512_256_Loss.png', dpi=1000, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

plt.show()


# In[29]:


#loss is the worst with high batch size (256). It is fastest, but batch_size of 64 shows best loss and accuracy,
#just 20 seconds longer.

#the smaller number of Nodes (256) is overall much faster than the greater number (512).

test_loss10, test_acc10 = model10.evaluate(X_test, y_test)
test_loss11, test_acc11 = model11.evaluate(X_test, y_test)
test_loss12, test_acc12 = model12.evaluate(X_test, y_test)
test_loss13, test_acc13 = model13.evaluate(X_test, y_test)
test_loss14, test_acc14 = model14.evaluate(X_test, y_test)
test_loss15, test_acc15 = model15.evaluate(X_test, y_test)


# In[30]:


#test_acc10 #0.6
#test_acc11 #0.5375
#test_acc12 #0.6
#test_acc13 #0.595
#test_acc14 #0.555
#test_acc15 #0.5325


# In[37]:


#add in a Conv2D (64) and maxPooling2D ((2, 2))

model20 = models.Sequential()
model20.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model20.add(layers.MaxPooling2D((2, 2)))
model20.add(layers.Flatten())
model20.add(layers.Dense(256, activation='relu'))
model20.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model20.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model20.fit(X_train,
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
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 64 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 64 btch)')
plt.legend()

plt.show()


# In[39]:


model21 = models.Sequential()
model21.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model21.add(layers.MaxPooling2D((2, 2)))
model21.add(layers.Flatten())
model21.add(layers.Dense(256, activation='relu'))
model21.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model21.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model21.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128,
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
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 128 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 128 btch)')
plt.legend()

plt.show()


# In[41]:


model22 = models.Sequential()
model22.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model22.add(layers.MaxPooling2D((2, 2)))
model22.add(layers.Flatten())
model22.add(layers.Dense(256, activation='relu'))
model22.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model22.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model22.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=256,
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
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 256 Nds, 256 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 256 Nds, 256 btch)')
plt.legend()

plt.show()


# In[ ]:


#512 nodes in dense layer


# In[43]:


#add in a Conv2D (64) and maxPooling2D ((2, 2))

model23 = models.Sequential()
model23.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model23.add(layers.MaxPooling2D((2, 2)))
model23.add(layers.Flatten())
model23.add(layers.Dense(512, activation='relu'))
model23.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model23.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model23.fit(X_train,
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
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 64 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 64 btch)')
plt.legend()

plt.show()


# In[45]:


model24 = models.Sequential()
model24.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model24.add(layers.MaxPooling2D((2, 2)))
model24.add(layers.Flatten())
model24.add(layers.Dense(512, activation='relu'))
model24.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model24.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model24.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[46]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 128 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 128 btch)')
plt.legend()

plt.show()


# In[47]:


model25 = models.Sequential()
model25.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(64, 64, 1)))
model25.add(layers.MaxPooling2D((2, 2)))
model25.add(layers.Flatten())
model25.add(layers.Dense(512, activation='relu'))
model25.add(layers.Dense(1, activation='sigmoid'))

##########################################################

model25.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

##########################################################

processing_time = []
    
start_time = time.clock()

history = model25.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=256,
                    validation_data=(X_val, y_val))

end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time 
print("\nProcessing time (seconds): %f" % runtime)        
processing_time.append(runtime)


# In[48]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Train & Valid Acc. (1 Conv/MaxPool lay, 512 Nds, 256 btch)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train & Valid Loss (1 Conv/MaxPool lay, 512 Nds, 256 btch)')
plt.legend()

plt.show()


# In[57]:


test_loss20, test_acc20 = model20.evaluate(X_test, y_test)
test_loss21, test_acc21 = model21.evaluate(X_test, y_test)
test_loss22, test_acc22 = model22.evaluate(X_test, y_test)
test_loss23, test_acc23 = model23.evaluate(X_test, y_test)
test_loss24, test_acc24 = model24.evaluate(X_test, y_test)
test_loss25, test_acc25 = model25.evaluate(X_test, y_test)

