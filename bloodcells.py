#blood cells


import os
os.environ['KERAS_BACKEND']="tensorflow"  
from sklearn.metrics import classification_report,roc_auc_score,roc_curve, confusion_matrix
from sklearn.model_selection import TunedThresholdClassifierCV    
from sklearn.metrics import make_scorer, f1_score,auc
from sklearn.preprocessing import label_binarize
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


path = kagglehub.dataset_download("unclesamulus/blood-cells-image-dataset")

print("Path to dataset files:", path)
dataset_dir=os.path.join(path,'bloodcells_dataset')


train_dataset=tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                          validation_split=0.2,
                                                          subset='training',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          seed=123,
                                                          label_mode='int'    #categorical
                                                          )

val_dataset = tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                          validation_split=0.2,
                                                          subset='validation',
                                                          image_size=(224,224),
                                                          batch_size=32,
                                                          label_mode='int',
                                                          seed=123)
test_dataset=tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                               
                                                            shuffle=False,
                                                            image_size=(224,224),
                                                            batch_size=32)

class_names=train_dataset.class_names
print("Class names:", class_names)
numclasses=len(class_names)

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



model = tf.keras.models.Sequential([      
    base_model,
    tf.keras.layers.Dense(units=256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(units=128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(numclasses,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
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
model.fit(train_dataset,shuffle=True,verbose=1, validation_data=val_dataset, epochs=0, callbacks=[lr_scheduler])



base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(train_dataset, validation_data=val_dataset, epochs=0,shuffle=True,verbose=1)




loss,accuracy = model.evaluate(val_dataset)
print("ACCURACY:",accuracy)
print("LOSS:",loss)

batch_size=32

y_true=np.concatenate([y for x, y in test_dataset]) 
test_dataset_for_pred=test_dataset.unbatch().batch(batch_size)  
predicted_results=model.predict(test_dataset_for_pred)

y_prob=predicted_results
predicted_results=np.argmax(predicted_results,axis=1)

print("Predicted results:",predicted_results)
print("classification report:\n", classification_report(y_true,y_pred=predicted_results,target_names=class_names))


y_true_oh = label_binarize(y_true, classes=list(range(numclasses)))

fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(10, 8))
for i in range(numclasses):  
    fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

#configurable parameters
model_builder=keras.applications.resnet.ResNet50
img_size=(224,224)
preprocess_input=keras.applications.resnet.preprocess_input
decode_predictions=keras.applications.resnet.decode_predictions


last_conv_layer_name = "conv5_block3_out"

img_path=keras.utils.get_file( "bloodcell_sample.jpg", "https://storage.googleapis.com/kagglesdsdata/datasets/2277635/3865189/bloodcells_dataset/erythroblast/ERB_115039.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251129%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251129T043614Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=47a0fd89c9ec782cf144504bc0c44d698ff195c9b2b81852c287d053ffb4ec7f9738e21f5865a093f0f71d4d908d0fe4ff797dedc16ca04b6f312680cf8c52d04025640c78fc0946e127f48b1c92864766ecf98b9d5ab4260d76fd611c8021e88abee83d08fdbc9b3f0871de31dd194a33fe1e4d386798bb18bc29f503d43a4f38f2e6f63ff017f2da6d36df5a6dcadd7211cee6857c4e3c261cf8a9f08bb68f17228cd938e280140f8eafcc9b558483c744972583ef641225727736af3555f78517a6b7b3ebb0856a23d4c110c361bbca7e88dc43ba9b470fe12904b4dbe33671452a3e559ff9217b1c371127e99cbb8ad370b014f3a41e2509c564a1aab3d1")
display(Image(img_path))
image = PILImage.open(img_path)
img = image.resize((224, 224))   #preprocessing the image
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
class_labels = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
score = tf.nn.softmax(predictions[0])
print(f"{class_labels[tf.argmax(score)]}")
plt.figure(figsize=(5,5))
plt.title(f"Predicted: {class_labels[tf.argmax(score)]} with {100 * tf.reduce_max(score):.2f} percent confidence")
plt.axis("off")
plt.show()

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
imagenet_model=model_builder(weights='imagenet',include_top=True)
imagenet_model.layers[-1].activation=None
preds=imagenet_model.predict(img_array)
print('Predicted:',decode_predictions(preds,top=1)[0])
heatmap=make_gradcam_heatmap(img_array,imagenet_model,last_conv_layer_name)
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()
save_and_display_gradcam(img_path,heatmap, cam_path="gradcam_overlay.jpg")



