"""
Date : unknown
Who : S.W. Leem

<24.08.20>
model 생성 코드
문제는 학습용 img가 존재하지 않는다.

in >> 20 x 20  sth
out << 2 values?

"""


#%% 0. ==============================================================================================
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras import layers, initializers, regularizers, losses
from keras.layers import LeakyReLU, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical, normalize
from keras import backend as K


#%% 1. Evaluation 지표 ==============================================================================================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # K.clip : Element-wise value clipping.
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%% 2. 모폴로지 적용 CNN ==============================================================================================

# (1) true
# img_dir = "C:\Users\dlatj\Anaconda3\chest_xray\train"
img_dir = 'C:/Users/dlatj/Anaconda3/floodmap/True2_n'  # Enter Directory of all images
data_path = os.path.join(img_dir, '*jpeg')   
files = glob.glob(data_path)    # unix style 로 path 를 찾기 위함 - 여기서는 asterisk 사용위해
true = []  # empty array for true data loading
for f1 in files:
    img = Image.open(f1)
    img = np.array(img)
    true.append(img)  # the image files are stored in true variable

# (2) false
# img_dir = "C:\Users\dlatj\Anaconda3\chest_xray\train"
img_dir = 'C:/Users/dlatj/Anaconda3/floodmap/False2_n'  # Enter Directory of all images
data_path = os.path.join(img_dir, '*jpeg')
files = glob.glob(data_path)
false = []  # empty array for false data loading
for f1 in files:
    img = Image.open(f1)
    img = np.array(img)
    false.append(img)  # the image files are stored in false variable



# (3) 개수 정리
n_true = np.shape(true)[0]  # number of true image
label_t = np.ones(n_true, dtype=int)  # label for one hot coding
n_false = np.shape(false)[0]  # number of false image
label_f = np.zeros(n_false, dtype=int)  # false label



# (4) putting labels and images in one array
image = []
labels = []
image.extend(true)  # 내부 원소를 새 원소로 이어주기
image.extend(false)
labels.extend(label_t)
labels.extend(label_f)
n_images = np.shape(image)[0]  # number of total images



# (5) normalization and labeling of the training data using the keras util normalization
image = normalize(image, axis=-1, order=2)
labels = to_categorical(labels, 2)  # binary class 로 classs vector 로 전환, labels 는 위에서 one-hot coding 된거
image = image.reshape(n_images, 20, 20, 1)  # reshaping the image to a format for CNN application


# (6) saving indices of training and test data
Train_test_split = 0.9  # splitting training and test data with ratio of 9:1
split_index = int(Train_test_split * n_images)
shuffled_indicies = np.random.permutation(n_images)  # 내부 array 를 random 한 차례로 섞어줌
train_indicies = shuffled_indicies[0:split_index]
test_indicies = shuffled_indicies[split_index:]

train_image2 = image[train_indicies, :]
test_image2 = image[test_indicies, :]
train_labels2 = labels[train_indicies]
test_labels2 = labels[test_indicies]


#%% 3. build model ==============================================================================================
# Keras Model for CNN application
"""
model2 = tf.keras.Sequential()
model2.add(Conv2D(filters=64, 
                  kernel_size=(5, 5), 
                  input_shape=(20, 20, 1),
                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='relu'))  # Initializer that generates tensors with a normal distribution.
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=32, 
                  kernel_size=(3, 3),
                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=16, 
                  kernel_size=(1, 1),
                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='relu'))
model2.add(MaxPooling2D(pool_size=1))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(layers.Dense(8, 
                        kernel_regularizer=regularizers.l2(0.01), 
                        activation='relu'))
model2.add(layers.Dense(2,
                        kernel_regularizer=regularizers.l2(0.01), 
                        activation='sigmoid'))
model2.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.75, decay=0.0, nesterov=False),  # optimizer : 보폭을 결정하는 느낌, SGD : gradient descent
               loss=losses.binary_crossentropy,
               metrics=['accuracy'])


#*model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, epsilon=1e-8, beta_1=0.9, beta_2=0.999, decay=0.002),
#*               loss='binary_crossentropy',
#*               metrics=['accuracy'])

"""
inputs = tf. keras.Input(shape=(20, 20, 1))
x = Conv2D(filters=64, 
           kernel_size=(5, 5), 
           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
           activation='relu')(inputs)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=32, 
           kernel_size=(3, 3), 
           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
           activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64, 
           kernel_size=(1, 1), 
           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
           activation='relu')(x)
x = MaxPooling2D(pool_size=1)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(8,
          kernel_regularizer=regularizers.l2(0.01), 
          activation='relu')(x)
outputs = Dense(2,
          kernel_regularizer=regularizers.l2(0.01), 
          activation='sigmoid')(x)    # 0 ~ 1 사이의 normalized 된 x,y 좌표를 뿌려줌

model2 = tf.keras.Model(inputs, outputs, name="unknown")

model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, # optimizer : 보폭을 결정하는 느낌, SGD : gradient descent
                                                 momentum=0.75, 
                                                 weight_decay=0.0, 
                                                 nesterov=False),  
               loss=losses.binary_crossentropy,
               metrics=['accuracy'])

model2.summary()

#%% 4. train  ==============================================================================================
AI2 = model2.fit(train_image2, 
                 train_labels2, 
                 epochs=10,
                 validation_data=(test_image2, test_labels2))


#%% 5. plot the result ==============================================================================================
# plotting results of the Model

fig = plt.figure(num=1, figsize=(6, 3))
rows = 1
cols = 2
ax1 = fig.add_subplot(rows, cols, 1)
model2.summary()
plt.plot(AI2.history['acc'])       # history 는 자동적으로 모든 keras model에 적용됨.
plt.plot(AI2.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
ax2 = fig.add_subplot(rows, cols, 2)
plt.plot(AI2.history['loss'])
plt.plot(AI2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss, accuracy, f1_score, precision, recall = model2.evaluate(test_image2, test_labels2, verbose=0)

model2.save('Morphology_CNN.h5')
