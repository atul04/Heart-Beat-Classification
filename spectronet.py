# @Author: nilanjan
# @Date:   2018-11-20T19:34:48+05:30
# @Email:  nilanjandaw@gmail.com
# @Filename: spectronet.py
# @Last modified by:   nilanjan
# @Last modified time: 2018-11-22T23:09:30+05:30
# @Copyright: Nilanjan Daw

from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras import callbacks
import csv
import os
import shutil


def makeTrainDataset():
    base_dir = 'training/'
    normal = base_dir + "normal/"
    abnormal = base_dir + "abnormal/"
    with open('training.csv') as file:
        dataset = csv.reader(file, delimiter=',')
        for data in dataset:
            if data[1] == '-1':
                filename = data[0] + ".png"
                src = base_dir + filename
                if os.path.isfile(src):
                    dst = normal + filename
                    shutil.move(src, dst)
            elif data[1] == '1':
                filename = data[0] + ".png"
                src = base_dir + filename
                if os.path.isfile(src):
                    dst = abnormal + filename
                    shutil.move(src, dst)


def makeValidationDataset():
    base_dir = 'validation/'
    normal = base_dir + "normal/"
    abnormal = base_dir + "abnormal/"
    with open('validation.csv') as file:
        dataset = csv.reader(file, delimiter=',')
        for data in dataset:
            if data[1] == '-1':
                filename = data[0] + ".png"
                src = base_dir + filename
                if os.path.isfile(src):
                    dst = normal + filename
                    shutil.move(src, dst)
            elif data[1] == '1':
                filename = data[0] + ".png"
                src = base_dir + filename
                if os.path.isfile(src):
                    dst = abnormal + filename
                    shutil.move(src, dst)


def defineModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(240, 360, 3)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.04), metrics=['acc'])
    return model


print("Compiling model...")
model = defineModel()
print("making training dataset...")
makeTrainDataset()
print("making validation dataset")
makeValidationDataset()

train_dir = 'training'
validation_dir = 'validation'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(
                                        rescale=1./255,
                                        featurewise_center=True,
                                        featurewise_std_normalization=True
                                        )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(240, 360),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(240, 360),
    batch_size=20,
    class_mode='binary'
)

csv_logger = callbacks.CSVLogger('log.csv', append=True, separator=';')
filepath = "weights/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(
    filepath, monitor='val_acc', save_best_only=True, mode='max')
tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                    write_grads=False, write_images=True, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[csv_logger, tensorboard, checkpoint]
)

model.save('spectronet.h5')
