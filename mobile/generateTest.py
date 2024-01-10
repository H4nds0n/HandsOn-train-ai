import tensorflow as tf
from keras import layers, models, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

test_data_dir = f'data/test_resized28'
model = load_model("models/model30/best_model_0.keras", compile=False)
class_names = open("models/model30/labels.txt", "r").readlines()

class_labels = sorted([d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d))])

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

predictions = model.predict(test_generator)
y_true = test_generator.classes
y_pred = predictions.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()