from keras import applications
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove, listdir
import matplotlib.pyplot as plt
import itertools



from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


from keras.models import load_model


from keras import metrics

import keras
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
img_height = img_width = 224
channels = 3

train_path = 'traffic_data\\train'
test_path  ='traffic_data\\test'
valid_path = 'traffic_data\\valid'

train_batches= ImageDataGenerator().flow_from_directory(train_path,target_size=(img_height, img_width),classes=['red','green'],batch_size=20)
test_batches= ImageDataGenerator().flow_from_directory(test_path,target_size=(img_height, img_width),classes=['red','green'],batch_size=12)
valid_batches= ImageDataGenerator().flow_from_directory(valid_path,target_size=(img_height, img_width),classes=['red','green'],batch_size=20)

vgg16_model =keras.applications.vgg16.VGG16()
model =Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


model.summary()



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



for layer in model.layers:
    layer.trainable= False

model.add(Dense(2,activation='softmax'))
model.summary()
model.load_weights('traffic_weights_2.h5')
#model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit_generator(train_batches,steps_per_epoch=10,validation_data=valid_batches,validation_steps=10,epochs=5,verbose=1)
#model.save_weights('traffic_weights_2.h5')
test_imgs ,test_labels=next(test_batches)
test_labels =test_labels[:,0]

predictions =model.predict_generator(test_batches)
cm =confusion_matrix(test_labels,np.round(predictions[:,0]))

cm_plot_labels =['green','red']
plot_confusion_matrix(cm,cm_plot_labels)
plt.figure()
plt.show()