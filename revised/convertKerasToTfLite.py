import tensorflow as tf

ITERATION=33

model = tf.keras.models.load_model(f"models/model{ITERATION}/best_model_0.keras")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
]
tflite_model = converter.convert()


# Save the TensorFlow Lite model
with open(f'models/model{ITERATION}/asl_model_best1.tflite', 'wb') as f:
    f.write(tflite_model)
    