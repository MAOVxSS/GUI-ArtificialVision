from tkinter import Tk
from GUI.Home.home_ui import stop_gif_animation
from Utils.gui_utils import center_window


# Función que se ejecuta cuando se hace clic en el botón diccionario
def button_dictionary_click(window):
    stop_gif_animation(window)  # Detener la animación del GIF antes de cambiar de ventana

    # Importar la función necesaria para crear la ventana "dictionary"
    from GUI.Dictionary.dictionary_ui import create_dictionary_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    center_window(new_window)
    button_alphabet, button_vocabulary, button_go_back, images = create_dictionary_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()


# Función que se ejecuta cuando se hace clic en el botón lecciones
def button_lessons_click(window):
    stop_gif_animation(window)  # Detener la animación del GIF antes de cambiar de ventana

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


# Función que se ejecuta cuando se hace clic en el botón información
def button_information_click(window):
    stop_gif_animation(window)  # Detener la animación del GIF antes de cambiar de ventana

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


# Configurar la lógica de los botones de la ventana "home"
def setup_home_window_logic(window, button_dictionary, button_lessons, button_information):
    # Asignar la función correspondiente a cada botón
    button_dictionary.config(command=lambda: button_dictionary_click(window))
    button_lessons.config(command=lambda: button_lessons_click(window))
    button_information.config(command=lambda: button_information_click(window))
