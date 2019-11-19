import keras #Библиотека нейроных сетей (слоев)
from skilearn.datasets import load_digits
from skilearm.model_selection import train_test_split

import keras.utils
from keras.models import Sequantial
from keras.layers import Conv2D, Flatten, Dense

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.label, test_size=1/5)
img_height = 8
img_width = 8
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)/16
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)/16

x_train = keras.utils.to_categorical(y_train, 10) #3 - > [0,0,1,0,0,0,0,0,0,0]
x_test = keras.utils.to_categorical(y_test,10)

model = Sequential()
model.add(Conv2D(32, kernal = (3,3)), activation='relu', input_shape=(8,8,1))) #Conv2D сверточный слой
model.add(Conv2D(64, kernal = (3,3)), activation ='relu')
model.add(Flatten()) # Объединить все предыдущие результаты в один.
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd=SGD(lr=0.5)#0.01 - 0.1 Коэфицент скорости обучения

model.compile(loss=categorical_crossentropy, optimizer=sgd,metrics=['accuracy'])

model.fix(x_train, x_train, validation_data=(x_test, y_test), verbose=1, batch_size=16, epochs=120)#batch_size число картинок

model.save_weights('w.h5')
loss,accuracy=model,evaluate(x_test,y_test)
print(accuracy)