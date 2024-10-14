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

# rutas para las interfaces gráficas y sus elementos
assets_home_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Home"))
assets_information_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Information"))
assets_dictionary_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Dictionary"))
assets_dictionary_alphabet_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Dictionary", "Alphabet"))
assets_lessons_alphabet_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Lessons_Alphabet"))
assets_letters_information_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Letters_Information"))

