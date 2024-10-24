# ARCHIVO PARA ESTILOS Y FUNCIONES
from tkinter import Canvas, Tk

# Configuración de las ventanas principales
WINDOW_CONFIG = {
    "bg": "#369FD6",  # Puedes cambiar esto si los colores difieren entre las ventanas
    "height": 768,
    "width": 1366,
    "bd": 0,
    "highlightthickness": 0,
    "relief": "ridge"
}


# Función para limpiar la ventana y configurarla
def setup_window(window, background_color="#369FD6"):
    """Configura la ventana con los estilos comunes y un color de fondo específico."""
    # Limpiar la ventana actual
    for widget in window.winfo_children():
        widget.destroy()

    # Configurar la ventana
    window.configure(bg=background_color)


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
    from GUI.Home.home_logic import setup_home_window_logic

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    center_window(new_window)

    # Crear la interfaz "home"
    button_dictionary, button_information, button_lessons, images = create_home_window(new_window)

    # Configurar la lógica de la ventana "home"
    setup_home_window_logic(new_window, button_dictionary, button_lessons, button_information)

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
    from GUI.Dictionary.Alphabet.dictionary_alphabet_ui import create_alphabet_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_a, button_b, button_c, button_d, button_e, button_f, button_g, button_h,
     button_i, button_j, button_k, button_l, button_m, button_n, button_nn, button_o, button_p,
     button_q, button_r, button_s, button_t, button_u, button_v, button_w, button_x,
     button_y, button_z, button_go_back, button_go_home, images) = create_alphabet_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()

# Función para generar la ventana "Alfabeto"
def go_lessons_alphabet_window(window):
    from GUI.Lessons.Alphabet.lessons_alphabet_ui import create_lessons_alphabet_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    (button_image_home, button_image_go_back, button_image_random_lesson,
     button_image_complete_lesson, images) = create_lessons_alphabet_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()
