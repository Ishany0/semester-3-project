#disease diagnosis using ct scan images

import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import kagglehub
import numpy as np
import pandas as pd


# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)


data_dir=os.path.join(path,'chest_xray','train')

train_dataset=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                          validation_split=0.2,
                                                          subset='training',
                                                          seed=123,
                                                          image_size=(160,160),
                                                          batch_size=32
                                                          )

val_dataset=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        validation_split=0.2,
                                                        subset='validation',
                                                        seed=123,
                                                        image_size=(160,160),
                                                        batch_size=32
                                                     )
#using 160x160 instead of 256x256 and batch size=32 to reduce computational cost (and enhance speed), you lose very little accuracy for medical tasks since textures dominate, not object detail.


#Returns A Model instance

base_model=tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(160,160, 3),
    pooling='avg',
    name='ResNet50'
)
#include_top=False means that the fully-connected layer at the top of the network will not be included.
#    classifier_activation='softmax' is not needed since we are not including the top layer
#avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
#pre-training on ImageNet, using None in weights instead of 'imagenet' would initialize the model with random weights and require training from scratch
# #The name of the model (string).

#Note: each Keras Application expects a specific kind of input preprocessing.
#  For ResNet, call keras.applications.resnet.preprocess_input on your inputs before 
# passing them to the model. resnet.preprocess_input will convert the input images from RGB to BGR,
#  then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

#train_dataset = tf.keras.applications.resnet.preprocess_input(np.array(train_dataset))
#is not array so we cant use it directly


#using num_parallel_calls because loading images can be slow

AUTOTUNE=tf.data.AUTOTUNE

#add performance improvement

train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
#That converts RGB→BGR and zero-centers per ImageNet stats, which is essential for pre-trained ResNet.

train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
#cache() keeps data in memory after loaded off disk
#Otherwise, drop .cache() and keep only .prefetch() to avoid memory overflow.


#fine-tuning, avoids massive gradient computation.
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Dense(units=128,activation='relu'),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=2,activation='softmax') 
]) 
#softmax for multi-class classification
#flatten layer is inappropriate for resnet-level data,results in 196,608 input neurons per sample — extremely inefficient and meaningless for raw images.
#also we are using global average pooling in resnet so no need to flatten

'''
feedforward neural network
model.add(tf.keras.layers.Flatten(input_shape=(256,256,3)))  
model.add(tf.keras.layers.Dense(units=128,activation='relu'))  
model.add(tf.keras.layers.Dense(units=128,activation='relu')) 
model.add(tf.keras.layers.Dense(units=2,activation='softmax'))

'''


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 

model.fit(train_dataset,validation_data=val_dataset,epochs=2,validation_freq=1) 

pred_dataset=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        image_size=(160,160),
                                                        batch_size=5
                                                     )

pred_dataset = pred_dataset.map(lambda x, y : (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
pred_dataset= pred_dataset.cache().prefetch(buffer_size=AUTOTUNE)


loss,accuracy = model.evaluate(val_dataset)
print("ACCURACY:",accuracy)
print("LOSS:",loss)
predicted_results=model.predict(pred_dataset)
predicted_results=np.argmax(predicted_results,axis=1)
print("Predicted results:",predicted_results)
print("classification report:\n", classification_report(y_true=np.concatenate([y for x, y in pred_dataset], axis=0),y_pred=predicted_results,target_names=['NORMAL','PNEUMONIA']))
print("confusion matrix:\n",confusion_matrix(y_true=np.concatenate([y for x, y in pred_dataset], axis=0),y_pred=predicted_results))
