import tensorflow as tf
from keras import layers, models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, Callback, LambdaCallback, ReduceLROnPlateau
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 10

ITERATION=6

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.5
    return lr

# Define the directory containing the training data
train_data_dir = f'data/train_resized{1}'
test_data_dir = f'data/test_resized{1}'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("GPU not found")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Get a list of subdirectories (class labels)
class_labels = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))])

# Define the file path
labelsFilePath = f'models/model{ITERATION}/labels.txt'

# Ensure the directory exists
directory = os.path.dirname(labelsFilePath)
if not os.path.exists(directory):
    os.makedirs(directory)

# Write class labels with indices to a file
with open(labelsFilePath, 'w') as file:
    for index, label in enumerate(class_labels):
        file.write(f'{index} {label}\n')

# Build the model
data_augmentation = models.Sequential(
  [
    layers.RandomFlip(
        "horizontal",
        input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#accidentely had 7 classes
#1
# model = models.Sequential([
#     data_augmentation,
#     layers.Rescaling(1./255),
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.05),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

#2
# model = models.Sequential([
#     data_augmentation,
#     layers.Rescaling(1./255),
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.05),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

#3
# model = models.Sequential([
#     data_augmentation,
#     layers.Rescaling(1./255),
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.05),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

#4
#getting pretty good, epoch 4; accuracy at 49%, loss 95%
# model = models.Sequential([
#     data_augmentation,
#     # layers.Rescaling(1./255),
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.05),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

#5 => isn't working at all
# model = models.Sequential()
# # input layer
# # Block 1
# model.add(layers.Conv2D(32,3,activation='relu',padding='same',input_shape = (IMG_HEIGHT,IMG_WIDTH, 3)))
# model.add(layers.Conv2D(32,3,activation='relu',padding='same'))
# #model.add(BatchNormalization())
# model.add(layers.MaxPooling2D(padding='same'))
# model.add(layers.Dropout(0.2))

# # Block 2
# model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
# model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
# #model.add(BatchNormalization())
# model.add(layers.MaxPooling2D(padding='same'))
# model.add(layers.Dropout(0.3))

# #Block 3
# model.add(layers.Conv2D(128,3,activation='relu',padding='same'))
# model.add(layers.Conv2D(128,3,activation='relu',padding='same'))
# #model.add(BatchNormalization())
# #model.add(MaxPooling2D(padding='same'))
# model.add(layers.Dropout(0.4))

# # fully connected layer
# model.add(layers.Flatten())

# model.add(layers.Dense(512,activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(128,activation='relu'))
# model.add(layers.Dropout(0.3))

# # output layer
# model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# model = models.Sequential([

# ])

# Load VGG16 model and modify for ASL recognition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

model.summary()

early_stoping = EarlyStopping(monitor='val_loss', 
                              min_delta=0.001,
                              patience= 5,
                              restore_best_weights= True, 
                              verbose = 0)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', 
                                         patience = 2, 
                                         factor=0.5 , 
                                         verbose = 1)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define data generators for loading and augmenting data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

lr_callback = LearningRateScheduler(lr_schedule)
# checkpoint = ModelCheckpoint('asl_vgg16_best_weights.h5', save_best_only=True, monitor='val_accuracy', mode='max')


checkpoint = ModelCheckpoint(
    f'models/model{ITERATION}/best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)


from keras.callbacks import TensorBoard
import datetime

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
# model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator, callbacks=[checkpoint, early_stoping, reduce_learning_rate], verbose=1)
history = model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator, callbacks=[checkpoint, reduce_learning_rate], verbose=1)

error = pd.DataFrame(history.history)

plt.figure(figsize=(18,5),dpi=200)
sns.set_style('darkgrid')

plt.subplot(121)
plt.title('Cross Entropy Loss',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.plot(error['loss'])
plt.plot(error['val_loss'])

plt.subplot(122)
plt.title('Classification Accuracy',fontsize=15)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.plot(error['accuracy'])
plt.plot(error['val_accuracy'])

plt.show()


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# Save the model
model.save(f'models/model{ITERATION}/asl_model_keras.keras')


# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(f'models/model{ITERATION}/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)