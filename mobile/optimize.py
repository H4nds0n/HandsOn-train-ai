import tensorflow as tf

ITERATION = 1

model = tf.keras.models.load_model(f"./models/model{ITERATION}/in/asl_model_keras.keras")


# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(f"./models/model{ITERATION}/out/convertedModel.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
interpreter.save(f"./models/model{ITERATION}/out/asl_model.tflite")

