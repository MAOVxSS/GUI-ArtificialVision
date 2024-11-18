# ARCHIVO PARA RUTAS DE CONTENIDO EN EL PROYECTO
import os


"""
Rutas para los modelos generados y datos necesarios
"""

# ruta raiz
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ruta para acceder a los modelos generados
generated_models_path = os.path.abspath(os.path.join(root_dir, "Data", "Generated_Models"))

# ruta para acceder a los archivos h5 para las letras dinámicas
dynamic_model_converted_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Dynamic_Model_Data",
                                                       "Converted_Data"))

# ruta para acceder a los frames de las letras dinámicas
dynamic_model_frames_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Dynamic_Model_Data",
                                                       "Frame_Actions"))

# ruta para acceder a los key points de las letras estáticas
static_model_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Static_Model_Data",
                                                      "keypoints.csv"))

# ruta para acceder a las etiquetas de las letras estáticas
static_model_data_labels_path = os.path.abspath(os.path.join(root_dir, "Data", "Static_Model_Data",
                                                      "keypoints_labels.csv"))

# ruta para acceder a los frames de las frases comunes
phrases_model_frames_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Phrases_Model_Data",
                                                       "Frame_Actions"))

# ruta para acceder a los archivos h5 para las frases
phrases_model_converted_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Phrases_Model_Data",
                                                       "Converted_Data"))

# ruta para acceder al archivo JSON con las frases
phrases_model_json_data_path = os.path.abspath(os.path.join(root_dir, "Data", "Phrases_Model_Data",
                                                       "phrases.json"))


"""
Rutas para acceder a los recursos de las ventanas de las interfaces
"""
assets_home_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Home"))
assets_information_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Information"))
assets_dictionary_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Dictionary"))
assets_dictionary_alphabet_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Dictionary", "Alphabet"))
assets_lessons_alphabet_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Lessons_Alphabet"))
assets_letters_information_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Letters_Information"))
assets_camera_path = os.path.abspath(os.path.join(root_dir, "GUI", "Assets", "Camera"))

