import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Utils.paths import generated_models_path
from Utils.config import static_model_name, static_model_lite_name

def convert_to_tflite(
    keras_model_path,
    tflite_model_path,
    optimize_for_size=False,
    quantize=False,
    use_select_tf_ops=True
):
    """
    Convierte un modelo Keras a TFLite y agrega técnicas de depuración.
    Parametros:
    - keras_model_path: Ruta al modelo Keras (.h5 o SavedModel).
    - tflite_model_path: Ruta de salida para el .tflite.
    - optimize_for_size: Si True, aplicará optimizaciones para el tamaño del modelo.
    - quantize: Si True, aplicará una cuantización dinámica.
    - use_select_tf_ops: Si True, usa las operaciones selectas de TF para permitir ops no soportadas nativamente.
    """

    # Cargar el modelo Keras
    print(f"[INFO] Cargando modelo Keras desde: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    model.summary()

    # Información de depuración del modelo
    print("[DEBUG] Información sobre las capas del modelo:")
    for i, layer in enumerate(model.layers):
        print(f"  Capa {i}: {layer.name}, Tipo: {layer.__class__.__name__}, Salida: {layer.output_shape}")

    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"[DEBUG] Dimensión de entrada del modelo: {input_shape}")
    print(f"[DEBUG] Dimensión de salida del modelo: {output_shape}")

    # Configurar el convertidor TFLite
    print("[INFO] Iniciando conversión a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aplicar optimizaciones
    if optimize_for_size:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        print("[DEBUG] Optimizaciones para tamaño activadas.")

    if quantize:
        # Cuantización dinámica
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[DEBUG] Cuantización dinámica activada.")

    # Usar TF Select Ops para operaciones no soportadas nativamente
    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # Deshabilitar el experimental_lower_tensor_list_ops
        converter._experimental_lower_tensor_list_ops = False
        print("[DEBUG] Soporte para SELECT_TF_OPS activado y _experimental_lower_tensor_list_ops deshabilitado.")

    # Convertir el modelo a TFLite
    tflite_model = converter.convert()
    print("[INFO] Conversión a TFLite completada.")

    # Guardar el modelo TFLite
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[INFO] Modelo TFLite guardado en: {tflite_model_path}")

    # Depuración: probar inferencia
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

    # Crear datos de prueba aleatorios
    test_input_shape = input_details[0]['shape']
    test_data = np.random.randn(*test_input_shape).astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], test_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("[DEBUG] Inferencia de prueba completada. Salida:")
    print(output_data)

    print("[INFO] El modelo TFLite con SELECT_TF_OPS parece estar funcionando correctamente.")

if __name__ == "__main__":
    keras_model_path = os.path.join(generated_models_path, static_model_name)
    tflite_model_path = os.path.join(generated_models_path, static_model_lite_name)
    convert_to_tflite(
        keras_model_path=keras_model_path,
        tflite_model_path=tflite_model_path,
        optimize_for_size=False,
        quantize=False,
        use_select_tf_ops=True
    )
