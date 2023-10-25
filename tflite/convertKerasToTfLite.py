import tensorflow as tf

ITERATION=2

model = tf.keras.models.load_model(f"models/model{ITERATION}/asl_model_keras.keras")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(f'models/model{ITERATION}/asl_model2.tflite', 'wb') as f:
    f.write(tflite_model)