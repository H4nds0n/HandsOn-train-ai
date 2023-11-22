import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import os

# Define constants
IMG_HEIGHT = 300
IMG_WIDTH = 300
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 10

ITERATION=3

# Define the directory containing the training data
train_data_dir = f'data/train_resized{ITERATION}'
test_data_dir = f'data/test_resized{ITERATION}'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("GPU not found")

devices = tf.config.list_physical_devices()
print(devices)

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
model = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Define data generators for loading and augmenting data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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

# Train the model
model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator)

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