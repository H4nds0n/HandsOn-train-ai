
# import tensorflow as tf
# from keras import layers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import os

# # Define constants
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# NUM_CLASSES = 5
# BATCH_SIZE = 32
# NUM_EPOCHS = 10
# ITERATION = 50

# # Define the directory containing the training data
# train_data_dir = f'data/train_resized{40}'
# test_data_dir = f'data/test_resized{40}'

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("GPU not found")

# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # Get a list of subdirectories (class labels)
# class_labels = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))])

# # Define the file path
# labelsFilePath = f'models/model{ITERATION}/labels.txt'

# # Ensure the directory exists
# directory = os.path.dirname(labelsFilePath)
# if not os.path.exists(directory):
#     os.makedirs(directory)

# # Write class labels with indices to a file
# with open(labelsFilePath, 'w') as file:
#     for index, label in enumerate(class_labels):
#         file.write(f'{label}\n')

# pretrainedModel = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )

# pretrainedModel.trainable = False

# inputs = pretrainedModel.input

# # x = tf.keras.layers.Flatten()(pretrainedModel.output)
# # x = tf.keras.layers.Dense(64, activation='relu')(pretrainedModel.output)  # Reduced the units to 64
# outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(pretrainedModel.output)

# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# adam = tf.keras.optimizers.Adam(
#     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
#     name='Adam'
# )

# model.compile(
#     optimizer=adam,
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.001,
#     patience=5,
#     restore_best_weights=True,
#     verbose=0
# )

# reduce_learning_rate = ReduceLROnPlateau(
#     monitor='val_accuracy',
#     patience=2,
#     factor=0.5,
#     verbose=1
# )

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='sparse'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='sparse'
# )

# checkpoint = ModelCheckpoint(
#     f'models/model{ITERATION}/best_model.keras',
#     save_best_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     verbose=1
# )

# model.fit(
#     train_generator,
#     epochs=NUM_EPOCHS,
#     validation_data=test_generator,
#     callbacks=[checkpoint, early_stopping, reduce_learning_rate],
#     verbose=1
# )

# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f'Test accuracy: {test_accuracy}')

# model.save(f'models/model{ITERATION}/asl_model_keras.keras')

# # Convert the model to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# tflite_model = converter.convert()

# with open(f'models/model{ITERATION}/asl_model.tflite', 'wb') as f:
#     f.write(tflite_model)





import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
ITERATION = 60

# Define the directory containing the training data
train_data_dir = f'data/train_resized{ITERATION}'
test_data_dir = f'data/test_resized{ITERATION}'

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
        file.write(f'{label}\n')

pretrainedModel = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrainedModel.trainable = False

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.layers.Resizing(128, 128)(inputs) #96 at #51
x = pretrainedModel(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)  # Reduced the units to 64
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)

model.compile(
    optimizer=adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
    verbose=0
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    factor=0.5,
    verbose=1
)

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

checkpoint = ModelCheckpoint(
    f'models/model{ITERATION}/best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping, reduce_learning_rate],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

model.save(f'models/model{ITERATION}/asl_model_keras.keras')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open(f'models/model{ITERATION}/asl_model.tflite', 'wb') as f:
    f.write(tflite_model)
