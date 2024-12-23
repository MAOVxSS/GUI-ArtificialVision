# ARCHIVO PARA ESTILOS Y FUNCIONES
from tkinter import Canvas, Tk
import json
from tkinter import Toplevel, PhotoImage, Label
from GUI.Camera.Camera_Letters.camera_letters_model_logic import relative_to_assets_camera

# Ruta para la información de las letras y frases
from Utils.paths import assets_json_letters_info_path, assets_json_phrases_info_path

# Configuración de las ventanas principales
WINDOW_CONFIG = {
    "bg": "#369FD6",  # Puedes cambiar esto si los colores difieren entre las ventanas
    "height": 768,
    "width": 1366,
    "bd": 0,
    "highlightthickness": 0,
    "relief": "ridge"
}


# Función para limpiar la ventana, configurarla y cambiar el título
def setup_window(window, background_color="#369FD6", title=""):
    """
    Configura la ventana con los estilos comunes, un color de fondo específico
    y un título personalizado.
    """
    # Limpiar la ventana actual
    for widget in window.winfo_children():
        widget.destroy()

    # Configurar la ventana
    window.configure(bg=background_color)

    # Establecer el título de la ventana
    if title:
        window.title(title)
    else:
        window.title("Sistema de Reconocimiento de la Lengua de Señas Mexicana")

    icon_path = "GUI/Assets/Home/logo_tese.png"

    # Configurar el ícono
    if icon_path:
        try:
            icon = PhotoImage(file=icon_path)
            window.iconphoto(True, icon)
        except Exception as e:
            print(f"Error al cargar el ícono: {e}")



# Función para crear un canvas con estilos comunes
def create_common_canvas(window, **kwargs):
    config = WINDOW_CONFIG.copy()
    config.update(kwargs)  # Actualizar con cualquier parámetro adicional específico

    canvas = Canvas(window, **config)
    canvas.place(x=0, y=0)
    return canvas


# Estilo común de botones
BUTTON_COMMON_CONFIG = {
    "borderwidth": 0,
    "highlightthickness": 0,
    "relief": "flat"
}


# Función para centrar la ventana
def center_window(window, width=1366, height=768):
    # Obtener el tamaño de la pantalla
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calcular la posición x e y para centrar la ventana
    x_position = (screen_width // 2) - (width // 2)
    y_position = (screen_height // 2) - (height // 2)

    # Configurar la geometría de la ventana
    window.geometry(f"{width}x{height}+{x_position}+{y_position}")


# FUNCIONES PARA IR A LAS VENTANAS ****************************************************************************

# Función para generar la ventana "Inicio"
def go_home_window(window):
    # Funciones para crear la ventana
    from GUI.Home.home_ui import create_home_window

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_dictionary, button_information, button_lessons, images = create_home_window(new_window)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Diccionario"
def go_dictionary_window(window):
    # Funciones para crear la ventana
    from GUI.Dictionary.dictionary_ui import create_dictionary_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    button_alphabet, button_vocabulary, button_go_back, images = create_dictionary_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Lecciones"
def go_lessons_window(window):
    # Importar la función necesaria para crear la ventana "dictionary"
    from GUI.Lessons.lessons_ui import create_lessons_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    button_alphabet, button_vocabulary, button_go_back, images = create_lessons_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Alfabeto"
def go_dictionary_alphabet_window(window):
    from GUI.Dictionary.Alphabet.dictionary_alphabet_ui import create_dictionary_alphabet_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_a, button_b, button_c, button_d, button_e, button_f, button_g, button_h,
     button_i, button_j, button_k, button_l, button_m, button_n, button_nn, button_o, button_p,
     button_q, button_r, button_s, button_t, button_u, button_v, button_w, button_x,
     button_y, button_z, button_go_back, button_go_home, images) = create_dictionary_alphabet_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Frases comunes"
def go_dictionary_phrases_window(window):
    from GUI.Dictionary.Phrases.dictionary_phrases_ui import create_dictionary_phrases_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_phrase_hello, button_phrase_bye, button_phrase_how_are_you, button_phrase_take_care,
     button_phrase_neutral, button_phrase_thank_you, button_phrase_please, button_phrase_you_are_welcome,
     button_go_back, button_go_home, images) = create_dictionary_phrases_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Lecciónes Alfabeto"
def go_lessons_alphabet_window(window):
    from GUI.Lessons.Alphabet.lessons_alphabet_ui import create_alphabet_lessons_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_image_home, button_image_go_back, button_image_random_lesson,
     button_image_complete_lesson, images) = create_alphabet_lessons_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Lecciónes Alfabeto"
def go_phrases_lessons_window(window):
    from GUI.Lessons.Phrases.phrases_lessons_ui import create_phrases_lessons_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_image_home, button_image_go_back, button_image_random_lesson,
     button_image_complete_lesson, images) = create_phrases_lessons_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana Camara en su módulo de letras
def go_camera_letters_window(window, actual_letter):
    from GUI.Camera.Camera_Letters.camera_letters_logic import update_icon_letter
    # Se guarda la letra actual para mantenerla en las ventanas
    actual_letter = actual_letter

    # Cargar el archivo JSON
    with open(assets_json_letters_info_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if actual_letter in data:
        letter_data = data[actual_letter]

        # Funciones para crear la ventana
        from GUI.Camera.Camera_Letters.camera_letters_ui import create_camera_letters_window

        # Destruir la ventana actual (Information)
        window.destroy()

        # Crear una nueva ventana para Home
        new_window = Tk()
        center_window(new_window)

        # Crear la interfaz "home"
        button_tip, button_go_back, button_go_home, images = create_camera_letters_window(new_window, actual_letter,
                                                                                          letter_data["movement"])

        update_icon_letter(new_window, letter_data["icon_path"])

        # Iniciar el loop principal para la nueva ventana
        new_window.mainloop()
    else:
        print(f"Letra '{actual_letter}' no encontrada en el archivo JSON")


# Función para generar la ventana Camara en su módulo de letras
def go_camera_phrases_window(window, actual_phrase):
    from GUI.Camera.Camera_Phrases.camera_phrases_logic import update_icon_phrase
    # Se guarda la letra actual para mantenerla en las ventanas
    actual_phrase = actual_phrase

    # Cargar el archivo JSON
    with open(assets_json_phrases_info_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if actual_phrase in data:
        phrase_data = data[actual_phrase]

        # Funciones para crear la ventana
        from GUI.Camera.Camera_Phrases.camera_phrases_ui import create_camera_phrases_window

        # Destruir la ventana actual (Information)
        window.destroy()

        # Crear una nueva ventana para Home
        new_window = Tk()
        center_window(new_window)

        # Crear la interfaz "home"
        button_tip, button_go_back, button_go_home, images = create_camera_phrases_window(new_window,
                                                                                          phrase_data[
                                                                                              "complete_phrase"])

        update_icon_phrase(new_window, phrase_data["icon_path"])

        # Iniciar el loop principal para la nueva ventana
        new_window.mainloop()
    else:
        print(f"Letra '{actual_phrase}' no encontrada en el archivo JSON")


# Función que se ejecuta cuando se hace clic en el botón información
def go_information_window(window):
    # Importar la función necesaria para crear la ventana "information"
    from GUI.Information.information_ui import create_information_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "information"
    new_window = Tk()
    center_window(new_window)
    button_go_back, images = create_information_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Lección Completa Alfabeto"
def go_complete_lesson_alphabet(window):
    # Funciones para crear la ventana
    from GUI.Lessons.Alphabet.Alphabet_Complete_Lesson.alphabet_complete_lesson_ui import (
        create_complete_lesson_alphabet_window)

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_tip, button_go_back, button_go_home, images = create_complete_lesson_alphabet_window(new_window)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()


# Función para generar la ventana "Lección Al Azar del Alfabeto"
def go_mix_lesson_alphabet(window):
    # Funciones para crear la ventana
    from GUI.Lessons.Alphabet.Alphabet_Mix_Lesson.alphabet_mix_lesson_ui import (
        create_alphabet_mix_lesson_window)

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_tip, button_go_back, button_go_home, images = create_alphabet_mix_lesson_window(new_window)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()

# Función para generar la ventana "Lección Al Azar del Alfabeto"
def go_complete_lesson_phrases(window):
    # Funciones para crear la ventana
    from GUI.Lessons.Phrases.Phrases_Complete_Lesson.phrases_complete_lesson_ui import create_complete_lesson_phrases_window

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_tip, button_go_back, button_go_home, images = create_complete_lesson_phrases_window(new_window)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()

# Función para generar la ventana "Lección Al Azar del Alfabeto"
def go_mix_lesson_phrases(window):
    # Funciones para crear la ventana
    from GUI.Lessons.Phrases.Phrases_Mix_Lesson.phrases_mix_lesson_ui import create_mix_lesson_phrases_window

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_tip, button_go_back, button_go_home, images = create_mix_lesson_phrases_window(new_window)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()


def show_completion_popup(window, progress_label, restart_lesson_callback):
    """Muestra una ventana emergente de éxito al completar la lección.

    Argumentos:
    window -- ventana principal de la aplicación.
    progress_label -- etiqueta de progreso de la lección.
    restart_lesson_callback -- función a ejecutar para reiniciar la lección.
    """
    # Crear la ventana emergente centrada en la pantalla
    popup = Toplevel(window)
    popup.title("¡Completado!")
    window_width, window_height = 665, 665
    screen_width, screen_height = popup.winfo_screenwidth(), popup.winfo_screenheight()
    x_position, y_position = (screen_width // 2) - (window_width // 2), (screen_height // 2) - (window_height // 2)
    popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    popup.resizable(False, False)

    # Cargar la imagen de éxito y asignarla a la ventana emergente
    completion_image = PhotoImage(file=relative_to_assets_camera("mensaje_exito.png"))
    label = Label(popup, image=completion_image)
    label.pack(padx=10, pady=10)
    popup.completion_image = completion_image  # Mantener la referencia a la imagen para evitar recolección de basura

    # Configurar el cierre automático de la ventana emergente después de 3 segundos y reiniciar la lección
    popup.after(3000, lambda: [popup.destroy(), restart_lesson_callback()])
    popup.transient(window)  # Establecer la ventana emergente como hija de la ventana principal
    popup.grab_set()  # Bloquear interacción con la ventana principal mientras la emergente esté activa