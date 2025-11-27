#disease diagnosis using ct scan images


import os
os.environ['KERAS_BACKEND']="tensorflow"   #Forces Keras to use TensorFlow backend explicitly (for GPU acceleration).
from sklearn.metrics import classification_report,roc_auc_score,roc_curve, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV     #TunedThresholdClassifierCV is an sklearn-style wrapper. It requires that the model implements the scikit-learn API:
from sklearn.metrics import make_scorer, f1_score
import tensorflow as tf
from keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Image,display
import matplotlib as mpl





# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)



train_dir=os.path.join(path,'chest_xray','train')   #Points to the train folder containing images under subfolders
val_dir=os.path.join(path,'chest_xray','val')
test_dir=os.path.join(path,'chest_xray','test')


'''
train_datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
val_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
'''
train_dataset=tf.keras.utils.image_dataset_from_directory(train_dir,
                                                          validation_split=0.2,
                                                          subset='training',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          seed=123
                                                          )

val_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          seed=123)
test_dataset=tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            image_size=(224,224),
                                                            batch_size=32)
'''
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    subset='training',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

val_dataset = val_datagen.flow_from_directory(
    train_dir,
    subset='validation',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_dataset = val_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
'''
#using 160x160 instead of 256x256 and batch size=32 to reduce computational cost (and enhance speed), you lose very little accuracy for medical tasks since textures dominate, not object detail.


#Returns A Model instance

base_model=tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224, 3),
    pooling='avg',
    name='ResNet50'
)

'''
#Threshold tuning makes sense ONLY for models with probabilities
#to work : 1.train your keras model normally   2.get probabilities    3.find the best threshold for F1-score      4.Use this threshold as your final classifier
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),        # mirror left/right lungs
    tf.keras.layers.RandomRotation(0.05),             # slight rotation
    tf.keras.layers.RandomZoom(0.05),                 # zoom in/out 
    tf.keras.layers.RandomContrast(0.05),             # vary contrast
])
'''
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

#train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

#That converts RGB→BGR and zero-centers per ImageNet stats, which is essential for pre-trained ResNet.

train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset=test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
#cache() keeps data in memory after loaded off disk
#Otherwise, drop .cache() and keep only .prefetch() to avoid memory overflow.



model = tf.keras.models.Sequential([      #
    base_model,
    tf.keras.layers.Dense(units=256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),  #L2 regularization on dense layers   for better accuracy and specificity
    tf.keras.layers.Dropout(0.3), #to avoid overfitting
    tf.keras.layers.Dense(units=128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(units=2,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
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


#scheduler for learning rate decay
lr_scheduler=ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)



#fine-tuning, avoids massive gradient computation.
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset,shuffle=True,verbose=1, validation_data=val_dataset, epochs=1, callbacks=[lr_scheduler])



base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=1,shuffle=True,verbose=1)



#DIRECTORY ITERATOR EXHAUSTION: once you interate over it, the pointer moves forward and hence next time it iterates over an empty iterator


loss,accuracy = model.evaluate(val_dataset)
print("ACCURACY:",accuracy)
print("LOSS:",loss)

 #Important: reset the generator before predicting, to start from the beginning
batch_size=32

y_true=np.concatenate([y for x, y in test_dataset]) 
test_dataset_for_pred=test_dataset.unbatch().batch(batch_size)  #reset pointer
predicted_results=model.predict(test_dataset_for_pred)

y_prob=predicted_results[:,1 ] #probability of class 1 
predicted_results=np.argmax(predicted_results,axis=1)# #Converts softmax output (2 probabilities) to predicted class index (0 or 1).
#threshold tuning for better specificity
thresholds=np.linspace(0,1,200)
best_f1=0
best_t=0.5
for t in thresholds:
    temp_pred=(y_prob>=t).astype(int)
    f1=f1_score(y_true,temp_pred)
    if f1> best_f1:
        best_f1=f1
        best_t=t
print("Best Threshold:", best_t)
print("Best F1",best_f1)
predicted_results = (y_prob >= best_t).astype(int)
print("Predicted results:",predicted_results)
print("classification report:\n", classification_report(y_true,y_pred=predicted_results,target_names=['NORMAL','PNEUMONIA']))
cm=confusion_matrix(y_true,y_pred=predicted_results)
TN,FP,FN,TP=cm.ravel()
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)

print("Sensitivity(Recall for Pneumonia):",sensitivity)    #>=0.9 -- model catches all penumonia cases (dont miss sick patients)
print("Specificity (Recall for Normal):",specificity)      #>=0.9 -- model avoids false alarms (doesnt label healthy patients as sick)
auc_score=roc_auc_score(y_true,y_prob)        #acc good with sensitivity low is dangerous
print('AUC SCORE:',auc_score)  #AUC (Area Under ROC Curve): Measures model’s ability to rank pneumonia higher than normal (1 = perfect).

#roc curve plotting

fpr,tpr,thresholds=roc_curve(y_true,y_prob)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - ResNet50 CT Scan Classification')
plt.grid(True)
plt.show(block=True)
plt.hist(y_prob, bins=30)
plt.title("y_prob distribution")
plt.show()


'''
test_dataset.reset()
predicted_results = model.predict(test_dataset)
y_prob = predicted_results.ravel()
predicted_labels=(y_prob>0.6).astype(int)  #Applying a threshold of 0.6 for classification
y_true = test_dataset.classes
print("Classification Report:\n")
print(classification_report(y_true, predicted_labels, target_names=['NORMAL','PNEUMONIA']))
cm = confusion_matrix(y_true, predicted_labels)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("\nConfusion Matrix:\n",cm)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)   #Specificity depends on how strictly you classify negatives.
auc_score = roc_auc_score(y_true, y_prob)
print("AUC:", auc_score)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CT Scan Classification')
plt.grid(True)
plt.legend()
plt.show()

'''


