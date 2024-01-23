import tensorflow as tf

ITERATION=60

model = tf.keras.models.load_model(f"models/model{ITERATION}/best_model.keras")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
]
tflite_model = converter.convert()


# Save the TensorFlow Lite model
with open(f'models/model{ITERATION}/asl_model_best4.tflite', 'wb') as f:
    f.write(tflite_model)
    

# import tensorflow as tf
# import tensorflow_model_optimization as tfmot

# ITERATION=40

# # Load the model
# model = tf.keras.models.load_model(f"models/model{ITERATION}/best_model.keras")

# # Define the pruning parameters
# pruning_params = {
#       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
#                                                                final_sparsity=0.80,
#                                                                begin_step=2000,
#                                                                end_step=4000)
# }

# # Apply pruning to the model
# model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# # Recompile the model after applying pruning
# model_for_pruning.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Fine-tune the model
# # model_for_pruning.fit(...)

# # Convert the pruned model to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] # enable TensorFlow Lite ops.

# # Set the number of threads to speed up execution of operators
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.representative_dataset = representative_dataset
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# tflite_model = converter.convert()

# # Save the TensorFlow Lite model
# with open(f'models/model{ITERATION}/asl_model_best1.tflite', 'wb') as f:
#     f.write(tflite_model)
