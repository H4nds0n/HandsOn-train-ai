import tensorflow as tf
#from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import os
# from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
# from tflite_model_maker import image_classifier

from tflite_model_maker.image_classifier import create as ImageClassifierCreate, ImageClassifier

from tflite_model_maker import ExportFormat

# Define constants
IMG_HEIGHT = 300
IMG_WIDTH = 300
NUM_CLASSES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 10

ITERATION=1

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
        file.write(f'{index} {label}\n')


# # Build the model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

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

#Building the model
#using EfficientNet Lite:
spec = model_spec.get('efficientnet_lite0')
model = ImageClassifierCreate(train_data=train_generator, model_spec=spec, epochs=NUM_EPOCHS)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'---Tests finished---\naccuracy: {test_accuracy} \nloss: {test_loss} \n---')

# Save the model
# model.save('models/model3/asl_model.keras')

#export model
model.export(export_dir=f'models/model{ITERATION}')