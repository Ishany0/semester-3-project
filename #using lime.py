#using lime
#local interpretable model-agnostic explanations
#lime focuses on explaining the model's prediction for individual instances

import os
os.environ['KERAS_BACKEND']="tensorflow"
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import kagglehub
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from lime import lime_image


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

'''
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 

model.fit(train_dataset,validation_data=val_dataset,epochs=2,validation_freq=1) 
'''
#lime
class_names = ['Pneumonia','Normal']
img_path=keras.utils.get_file( "img.jpg", "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251109T151718Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8efb9ce56ec3a0446e9043ff5d8ce3cbb93e7f145d48a32fe0f057f212fc9ad81890b200f5befc51b68dd45fefe03e18c3ab563d0f47742dfba30eab7ff845b372e552c93a6cdb16c87fd9a2a90cce6012fd440e7601c91f579252ee62d5229e11dbc507a4158573286460a02eab12e68be888736cc3958034bdcb364bcbf7f54c18b84b11586687c822acc55e2aa5c6efd2ed6bd4d32dd688d1f021ba009bf35c00e6c43f3c06b529f7697fe215b5c59425c700fcb5d0070c278f0d8b263de0f2476b6767d7fe2f04b3cf41c5076648803ac1de68e799f038563205197e538dc40bf7a39465169b9657a22c31a5b10baf38d9c7fa2c046616426768fe8b78be")
img=keras.preprocessing.image.load_img(img_path,target_size=(299,299))
img_array=keras.preprocessing.image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array=img_array.astype(np.float32)
explainer=lime_image.LimeImageExplainer()
explanation=explainer.explain_instance(
    image=img_array[0].astype(np.double),
    classifier_fn=lambda imgs: model.predict(preprocess_input(np.array(imgs))),
    top_labels=2,
    hide_color=0,
    num_samples=1000
)


#visualize explanation
from skimage.segmentation import mark_boundaries

temp,mask= explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    hide_rest=False
)
plt.imshow(mark_boundaries(temp/2 + 0.5, mask))
plt.title(f"LIME explanation for class: {class_names[explanation.top_labels[0]]}")
plt.axis('off')
plt.show()




'''
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

'''


