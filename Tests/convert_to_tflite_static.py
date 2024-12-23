import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Utils.paths import generated_models_path, static_model_test_data_path
from Utils.config import static_model_name, static_model_lite_name

def convert_to_tflite(
    keras_model_path,
    tflite_model_path,
    csv_path,
    optimize_for_size=False,
    quantize=False,
    use_select_tf_ops=True
):
    """
    Convierte un modelo Keras a TFLite con soporte para cuantización y dataset representativo.
    Parámetros:
    - keras_model_path: Ruta al modelo Keras (.h5 o SavedModel).
    - tflite_model_path: Ruta de salida para el modelo .tflite.
    - csv_path: Ruta al archivo CSV que contiene el dataset representativo.
    - optimize_for_size: Si True, optimiza el tamaño del modelo.
    - quantize: Si True, aplica cuantización dinámica.
    - use_select_tf_ops: Si True, usa operaciones SELECT_TF_OPS para compatibilidad extendida.
    """
    # Cargar el modelo Keras
    print(f"[INFO] Cargando modelo Keras desde: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    model.summary()

    # Cargar el dataset representativo desde el archivo CSV
    print(f"[INFO] Cargando dataset representativo desde: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

    # Separar identificadores (primera columna) y datos de la mano (resto de columnas)
    X_representative = data[:, 1:]  # Elimina la columna de identificadores

    # Generador para el dataset representativo
    def representative_dataset():
        for i in range(len(X_representative)):
            # Ajusta los datos a la forma esperada por el modelo (batch_size=1, input_dim=42)
            yield [X_representative[i].reshape(1, -1)]

    # Configurar el convertidor TFLite
    print("[INFO] Iniciando conversión a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aplicar optimizaciones
    if optimize_for_size:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        print("[DEBUG] Optimizaciones para tamaño activadas.")

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        print("[DEBUG] Cuantización dinámica con dataset representativo activada.")

    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True  # Permitir operaciones personalizadas
        print("[DEBUG] SELECT_TF_OPS activado para compatibilidad extendida.")

    # Convertir el modelo a TFLite
    tflite_model = converter.convert()
    print("[INFO] Conversión a TFLite completada.")

    # Guardar el modelo TFLite
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[INFO] Modelo TFLite guardado en: {tflite_model_path}")

    # Depuración: probar inferencia en TFLite
    print("[INFO] Iniciando pruebas de inferencia en el modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[DEBUG] Detalles de entrada del intérprete TFLite:")
    for inp in input_details:
        print(f"  Índice: {inp['index']}, Forma: {inp['shape']}, Tipo: {inp['dtype']}")

    print("[DEBUG] Detalles de salida del intérprete TFLite:")
    for out in output_details:
        print(f"  Índice: {out['index']}, Forma: {out['shape']}, Tipo: {out['dtype']}")

    # Crear datos de prueba a partir del dataset representativo
    test_data = X_representative[0].reshape(1, -1).astype(input_details[0]['dtype'])  # Tomar el primer ejemplo
    interpreter.set_tensor(input_details[0]['index'], test_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("[DEBUG] Inferencia de prueba completada. Salida:")
    print(output_data)

    print("[INFO] El modelo TFLite parece estar funcionando correctamente.")

if __name__ == "__main__":
    # Rutas de los modelos y dataset representativo
    keras_model_path = os.path.join(generated_models_path, static_model_name)
    tflite_model_path = os.path.join(generated_models_path, static_model_lite_name)
    csv_path = static_model_test_data_path

    # Convertir el modelo a TFLite con configuraciones específicas
    convert_to_tflite(
        keras_model_path=keras_model_path,
        tflite_model_path=tflite_model_path,
        csv_path=csv_path,
        optimize_for_size=True,  # Optimizar el tamaño del modelo
        quantize=True,  # Habilitar cuantización
        use_select_tf_ops=False  # No usar SELECT_TF_OPS si no es necesario
    )
