import tensorflow as tf
import os

# Rutas y variables
from Utils.paths import generated_models_path
from Utils.config import static_model_name

# Cargar el modelo .keras
model_path = os.path.join(generated_models_path, static_model_name)
model = tf.keras.models.load_model(model_path)

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido a un archivo .tflite
with open('static_keypoint_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite y guardado como modelo.tflite")
