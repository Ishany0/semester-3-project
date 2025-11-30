#covid 19 X-ray and ct scan classification


import os
os.environ['KERAS_BACKEND']="tensorflow"  
from sklearn.metrics import classification_report,roc_auc_score,roc_curve, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV    
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
from PIL import Image as PILImage



path =r"H:\COVID-19 Dataset"

print("Path to dataset files:", path)

train_dir1=os.path.join(path,"CT") 
#train_dir2=os.path.join(path,"X-Ray")
test_dir1=os.path.join(path,"TEST-CT")
#test_dir2=os.path.join(path,"TEST-X-Ray")


train_dataset=tf.keras.utils.image_dataset_from_directory(train_dir1,
                                                          validation_split=0.2,
                                                          subset='training',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          seed=123
                                                          )


val_dataset = tf.keras.utils.image_dataset_from_directory(train_dir1,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          seed=123)

test_dataset=tf.keras.utils.image_dataset_from_directory(test_dir1,
                                                            image_size=(224,224),
                                                            batch_size=32)


base_model=tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224, 3),
    pooling='avg',
    name='ResNet50'
)



AUTOTUNE=tf.data.AUTOTUNE

train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)


train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset=test_dataset.cache().prefetch(buffer_size=AUTOTUNE)



model = tf.keras.models.Sequential([      #
    base_model,
    tf.keras.layers.Dense(units=256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(units=128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(units=2,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
])

lr_scheduler=ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset,shuffle=True,verbose=1, validation_data=val_dataset, epochs=1, callbacks=[lr_scheduler])



base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=1,shuffle=True,verbose=1)



loss,accuracy = model.evaluate(val_dataset)
print("ACCURACY:",accuracy)
print("LOSS:",loss)

batch_size=32

y_true=np.concatenate([y for x, y in test_dataset]) 
test_dataset_for_pred=test_dataset.unbatch().batch(batch_size)  
predicted_results=model.predict(test_dataset_for_pred)

y_prob=predicted_results[:,1 ] 
predicted_results=np.argmax(predicted_results,axis=1)
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
print("classification report:\n", classification_report(y_true,y_pred=predicted_results,target_names=['COVID-19',"NON-COVID-19"]))
cm=confusion_matrix(y_true,y_pred=predicted_results)
TN,FP,FN,TP=cm.ravel()
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)

print("Sensitivity(Recall for COVID-19):",sensitivity) 
print("Specificity (Recall for Normal):",specificity)   
auc_score=roc_auc_score(y_true,y_prob)  
print('AUC SCORE:',auc_score) 

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



#configurable parameters

img_size=(224,224)


last_conv_layer_name = "conv5_block3_out"

img_path=r"H:\COVID-19 Dataset\CT\COVID\2020.02.22.20024927-p19-68%5.png"
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
preds=model.predict(img_array)
print("Model raw prediction:", preds)
print("Predicted class:", np.argmax(preds))
heatmap=make_gradcam_heatmap(img_array,base_model,last_conv_layer_name)
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()
save_and_display_gradcam(img_path,heatmap, cam_path="gradcam_overlay.jpg")


