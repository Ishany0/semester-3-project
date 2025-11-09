#using grad-cam

import os
os.environ['KERAS_BACKEND']="tensorflow"
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import kagglehub
import numpy as np
import pandas as pd
from IPython.display import Image,display
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras


# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)


data_dir=os.path.join(path,'chest_xray','train')

train_dataset=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                          validation_split=0.2,
                                                          subset='training',
                                                          seed=123,
                                                          image_size=(299,299),
                                                          batch_size=32
                                                          )

val_dataset=tf.keras.utils.image_dataset_from_directory(data_dir,
                                                        validation_split=0.2,
                                                        subset='validation',
                                                        seed=123,
                                                        image_size=(299,299),
                                                        batch_size=32
                                                     )

base_model=tf.keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(299,299, 3),
    pooling='avg',
    classes=1000,
    classifier_activation='softmax',
    name='Xception'
)

AUTOTUNE=tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Dense(units=128,activation='relu'),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=2,activation='softmax') 
]) 



#configurable parameters
model_builder=keras.applications.xception.Xception
img_size=(299,299)
#we are using xception here because it is known to work well with grad-cam
#also , xception expects 299x299 images
preprocess_input=keras.applications.xception.preprocess_input
decode_predictions=keras.applications.xception.decode_predictions

last_conv_layer_name="block14_sepconv2_act"

img_path=keras.utils.get_file( "img.jpg", "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251109T151718Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8efb9ce56ec3a0446e9043ff5d8ce3cbb93e7f145d48a32fe0f057f212fc9ad81890b200f5befc51b68dd45fefe03e18c3ab563d0f47742dfba30eab7ff845b372e552c93a6cdb16c87fd9a2a90cce6012fd440e7601c91f579252ee62d5229e11dbc507a4158573286460a02eab12e68be888736cc3958034bdcb364bcbf7f54c18b84b11586687c822acc55e2aa5c6efd2ed6bd4d32dd688d1f021ba009bf35c00e6c43f3c06b529f7697fe215b5c59425c700fcb5d0070c278f0d8b263de0f2476b6767d7fe2f04b3cf41c5076648803ac1de68e799f038563205197e538dc40bf7a39465169b9657a22c31a5b10baf38d9c7fa2c046616426768fe8b78be")
display(Image(img_path))

#the grad-cam algorithm
def get_img_array(img_path,size):
    img=keras.utils.load_img(img_path,target_size=size)
    array=keras.utils.img_to_array(img) #array
    array=np.expand_dims(array,axis=0) #to add a dimension
    return array
def make_gradcam_heatmap(img_array,model,last_conv_layer_name,pred_index=None):
    grad_model=keras.models.Model(model.inputs,[model.get_layer(last_conv_layer_name).output,model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output,preds = grad_model(img_array)
        if pred_index is None:
            pred_index=tf.argmax(preds[0])
        class_channel = preds[:,pred_index]
    grads=tape.gradient(class_channel,last_conv_layer_output)
    pooled_grads=tf.reduce_mean(grads,axis=(0,1,2))
    last_conv_layer_output=last_conv_layer_output[0]
    heatmap=last_conv_layer_output @ pooled_grads[...,tf.newaxis]
    heatmap=tf.squeeze(heatmap)
    heatmap=tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#create a superimposed visualization
def save_and_display_gradcam(img_path,heatmap,cam_path="cam.jpg",alpha=0.4):
    img=keras.utils.load_img(img_path)
    img=keras.utils.img_to_array(img)
    heatmap=np.uint8(255*heatmap)
    jet=mpl.colormaps['jet']
    jet_colors=jet(np.arange(256))[:,:3]
    jet_heatmap=jet_colors[heatmap]
    jet_heatmap=keras.utils.array_to_img(jet_heatmap)
    jet_heatmap=jet_heatmap.resize((img.shape[1],img.shape[0]))
    jet_heatmap=keras.utils.img_to_array(jet_heatmap)
    superimposed_img=jet_heatmap*alpha+img
    superimposed_img=keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))

#test-drive it
img_array=preprocess_input(get_img_array(img_path,size=img_size))
model=model_builder(weights='imagenet')
model.layers[-1].activation=None
preds=model.predict(img_array)
print('Predicted:',decode_predictions(preds,top=1)[0])
heatmap=make_gradcam_heatmap(img_array,model,last_conv_layer_name)
plt.matshow(heatmap)
plt.show()
save_and_display_gradcam(img_path,heatmap)



base_model=tf.keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(299,299, 3),
    pooling='avg',
    classes=1000,
    classifier_activation='softmax',
    name='Xception'
)

AUTOTUNE=tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x),y),num_parallel_calls=AUTOTUNE)
train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Dense(units=128,activation='relu'),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=2,activation='softmax') 
]) 


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
