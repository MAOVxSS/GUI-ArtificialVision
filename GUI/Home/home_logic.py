from tkinter import Tk
from GUI.Home.home_ui import stop_gif_animation

# Función que se ejecuta cuando se hace clic en el botón diccionario
def on_button_1_click(window):
    print("button_1 clicked")

    stop_gif_animation(window)  # Detener la animación del GIF antes de cambiar de ventana

    # Importar la función necesaria para crear la ventana "dictionary"
    from GUI.Dictionary.dictionary_ui import create_dictionary_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "dictionary"
    new_window = Tk()
    new_window.geometry("1366x768")
    button_1, button_2, button_3, images = create_dictionary_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()

# Función que se ejecuta cuando se hace clic en el botón lecciones
def on_button_2_click():
    print("button_2 clicked")

# Función que se ejecuta cuando se hace clic en el botón información
def on_button_3_click(window):
    print("button_3 clicked")
    stop_gif_animation(window)  # Detener la animación del GIF antes de cambiar de ventana

    # Importar la función necesaria para crear la ventana "information"
    from GUI.Information.information_ui import create_information_window

    # Destruir la ventana actual (Home)
    window.destroy()

    # Crear una nueva ventana para "information"
    new_window = Tk()
    new_window.geometry("1366x768")
    button_1, images = create_information_window(new_window)

    # Iniciar el bucle principal para la nueva ventana
    new_window.mainloop()

# Configurar la lógica de los botones de la ventana "home"
def setup_home_window_logic(window, button_1, button_2, button_3):
    # Asignar la función correspondiente a cada botón
    button_1.config(command=lambda: on_button_1_click(window))
    button_2.config(command=on_button_2_click)
    button_3.config(command=lambda: on_button_3_click(window))
