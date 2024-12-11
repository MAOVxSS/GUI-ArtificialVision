import tensorflow as tf
import os
from Utils.paths import phrases_model_keras_path

# Ruta del modelo Keras
keras_model_path = os.path.join(phrases_model_keras_path, "phrases_model.keras")

# Ruta para guardar el modelo TFLite
tflite_model_path = os.path.join(phrases_model_keras_path, "phrases_model.tflite")

# Cargar el modelo Keras
model = tf.keras.models.load_model(keras_model_path)

# Configurar el convertidor TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Habilitar las operaciones de TensorFlow seleccionadas
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS     # Operaciones seleccionadas de TensorFlow
]

# Configurar dimensiones de entrada estáticas
converter.experimental_new_converter = True  # Usar el convertidor TFLite más reciente
converter._experimental_lower_tensor_list_ops = False  # Evitar conflictos con TensorListReserve

# Realizar la conversión
tflite_model = converter.convert()

# Guardar el modelo convertido
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"[INFO] Modelo convertido y guardado en: {tflite_model_path}")
