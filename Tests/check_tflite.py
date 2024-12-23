import tensorflow as tf
import os
import numpy as np
from Utils.paths import phrases_model_keras_path
from Utils.config import phrases_model_lite_name

# Ruta al modelo TFLite
tflite_model_path = os.path.join(phrases_model_keras_path, phrases_model_lite_name)

print("=== Verificación de modelo en Raspberry Pi ===")
try:
    # Cargar el intérprete TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[INFO] Modelo cargado correctamente en Raspberry Pi.")
    print(f"[INFO] Detalles de entrada: {input_details}")
    print(f"[INFO] Detalles de salida: {output_details}")

    # Verificar tensores estáticos
    for detail in input_details:
        print(f"[INFO] Tensor de entrada: {detail['name']}")
        print(f" - Forma esperada: {detail['shape']}")
        print(f" - Tipo: {detail['dtype']}")

    for detail in output_details:
        print(f"[INFO] Tensor de salida: {detail['name']}")
        print(f" - Forma esperada: {detail['shape']}")
        print(f" - Tipo: {detail['dtype']}")

    # Crear entrada ficticia para verificar ejecución
    input_shape = input_details[0]['shape']
    test_input = np.random.random(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("[INFO] Modelo ejecutado correctamente en Raspberry Pi.")
    print(f"[INFO] Salida del modelo: {output_data}")

except Exception as e:
    print("[ERROR] No se pudo cargar o ejecutar el modelo TFLite.")
    print(f"Error: {e}")

print("=== Verificación finalizada ===")
