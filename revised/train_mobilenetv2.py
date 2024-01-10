import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, Callback, LambdaCallback, ReduceLROnPlateau
import os

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10

ITERATION=33

#epoch 1 ==> slow but good: accuracy: 0.9785; loss: 0.0744
# it probably was because I densed it to 29 but it should've been num_classes (10)

# Define the directory containing the training data
train_data_dir = f'data/train_resized{31}'
test_data_dir = f'data/test_resized{31}'

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



pretrainedModel = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
     include_top=False,
     weights='imagenet',
     pooling='avg'
)

pretrainedModel.trainable = False


inputs = pretrainedModel.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrainedModel.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

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

early_stoping = EarlyStopping(monitor='val_loss', 
                              min_delta=0.001,
                              patience= 5,
                              restore_best_weights= True, 
                              verbose = 0)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_accuracy', 
                                         patience = 2, 
                                         factor=0.5 , 
                                         verbose = 1)

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
checkIter = 0

def update_check_iter():
    checkIter += 1

checkpoint = ModelCheckpoint(
    f'models/model{ITERATION}/best_model_{checkIter}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    callbacks=[LambdaCallback(on_epoch_end=update_check_iter)]
)

# Train the model
# model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator, callbacks=[checkpoint, early_stoping, reduce_learning_rate], verbose=1)
model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator, callbacks=[checkpoint, early_stoping, reduce_learning_rate], verbose=1)

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