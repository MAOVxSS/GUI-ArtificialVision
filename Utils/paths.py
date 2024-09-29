# ARCHIVO PARA RUTAS DE CONTENIDO EN EL PROYECTO
import os

# ruta raiz
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# rutas para los modelos generados y datos necesarios
generated_models_path = os.path.abspath(os.path.join(root_dir, "Data", "Model_Keras"))
dynamic_model_converted_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Dynamic_Model_Data",
                                                       "Converted_Data"))
dynamic_model_unconverted_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Dynamic_Model_Data",
                                                       "Frame_Actions"))
static_model_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Static_Model_Data",
                                                      "keypoints.csv"))
static_model_data_labels_path = os.path.abspath(os.path.join(root_dir, "Data", "Static_Model_Data",
                                                      "keypoints_labels.csv"))

