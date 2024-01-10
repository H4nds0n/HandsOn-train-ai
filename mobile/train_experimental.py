import tensorflow as tf
from keras import layers, models, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define global constant
ITERATION = 26

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 2:
        lr *= 0.5
    return lr

def create_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes),
        layers.Activation("softmax")
    ])
    return model

def train_model():
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 5

    # Define the directory containing the training data based on the global constant
    train_data_dir = f'data/train_resized{ITERATION}'
    test_data_dir = f'data/test_resized{ITERATION}'

    # Create model
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)
    model.summary()

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
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        f'models/model{ITERATION}/best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Train the model
    model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=test_generator, callbacks=[lr_callback, checkpoint, early_stop], verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy}')

    # Confusion Matrix
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = predictions.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Save the model
    model.save(f'models/model{ITERATION}/asl_model_keras.keras')

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(f'models/model{ITERATION}/asl_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    train_model()
