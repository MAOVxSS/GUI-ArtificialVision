# ARCHIVO PARA CONFIGURACIONES

# Nombre del los modelos generados
dynamic_model_name = "dynamic_model.keras"
static_model_name = "static_keypoint_model.keras"
static_model_lite_name = "static_keypoint_model.tflite"
phrases_model_keras_name = "phrases_model.keras"
phrases_model_lite_name = "phrases_model.tflite"

# Camara
id_camera = 1

# Letras y frases dinámicas

phrases_to_text = {
    "j_izq": "j",
    "j_der": "j",
    "q_izq": "q",
    "q_der": "q",
    "x_izq": "x",
    "x_der": "x",
    "z_izq": "z",
    "z_der": "z",
    "nn_der": "nn",
    "nn_izq": "nn",
    "k_der": "k",
    "k_izq": "k",
    "hola_izq": "hola",
    "hola_der": "hola",
    "de_nada_izq": "de_nada",
    "de_nada_der": "de_nada",
    "adios_izq": "adios",
    "adios_der": "adios",
    "mas_o_menos_izq": "mas_o_menos",
    "mas_o_menos_der": "mas_o_menos",
    "gracias_izq": "gracias",
    "gracias_der": "gracias",
    "como_estas": "como_estas",
    "por_favor": "por_favor",
    "cuidate": "cuidate"
}

# Definición del alfabeto reconocido por el sistema de señas
PHRASES = ['COMO_ESTAS', 'POR_FAVOR', 'HOLA', 'DE_NADA', 'ADIOS',
           'CUIDATE', 'MAS_O_MENOS', "GRACIAS"]

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'NN', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z']



