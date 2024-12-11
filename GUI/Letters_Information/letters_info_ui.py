from pathlib import Path
from tkinter import Button, PhotoImage

# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Importación de la ruta hacia los recursos necesarios para la ventana
from Utils.paths import assets_letters_information_path


# Función que genera la ruta completa hacia los datos dentro de la carpeta de recursos
def relative_to_assets_letters_information(path: str) -> Path:
    return assets_letters_information_path / Path(path)


# Función para crear la ventana con la información de una letra específica
def create_letters_information_window(window, actual_letter):
    # Guardar la letra actual para mantener la referencia en la ventana
    actual_letter = actual_letter

    # Configurar la ventana principal con un color de fondo
    setup_window(window, background_color="#369FD6")

    # Crear un canvas común que actuará como contenedor de los elementos de la interfaz
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes para evitar que sean recolectadas por el recolector de basura
    images = {}

    # Crear un rectángulo para mostrar la imagen de la seña (extraída del JSON "sign_path")
    canvas.create_rectangle(
        609.0, 28.0, 1338.0, 529.0, fill="#233D4D", outline="")

    # Crear un rectángulo para contener el texto dinámico de la descripción ("description" del JSON)
    canvas.create_rectangle(
        28.0, 219.0, 558.0, 738.0, fill="#233D4D", outline="")

    # Crear un rectángulo decorativo para el encabezado del texto
    canvas.create_rectangle(
        44.0, 43.0, 320.0, 191.0, fill="#F6AA1C", outline="")

    # Botones de la interfaz

    # Botón de ayuda (muestra una ventana emergente con ayuda)
    button_image_help = PhotoImage(file=relative_to_assets_letters_information("pregunta.png"))
    images["pregunta"] = button_image_help
    button_help = Button(image=button_image_help, **BUTTON_COMMON_CONFIG)
    button_help.place(x=437.0, y=630.0, width=87.0, height=70.0)

    # Botón para regresar a la ventana anterior (diccionario del alfabeto)
    button_image_go_back = PhotoImage(file=relative_to_assets_letters_information("button_1.png"))
    images["button_1"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=598.0, y=567.0, width=154.0, height=144.0)

    # Botón para ir a la ventana principal (home)
    button_image_go_home = PhotoImage(file=relative_to_assets_letters_information("button_2.png"))
    images["button_2"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=757.0, y=565.0, width=203.0, height=146.0)

    # Botón para ir a la ventana de la cámara (para ver la seña en vivo)
    button_image_go_camera = PhotoImage(file=relative_to_assets_letters_information("button_3.png"))
    images["button_3"] = button_image_go_camera
    button_go_camera = Button(image=button_image_go_camera, **BUTTON_COMMON_CONFIG)
    button_go_camera.place(x=963.0, y=565.0, width=378.0, height=146.0)

    # Importar funciones necesarias para la navegación entre ventanas
    from GUI.gui_utils import go_dictionary_alphabet_window
    from GUI.Letters_Information.letters_info_logic import stop_letter_gif_animation
    from GUI.gui_utils import go_home_window
    from GUI.Letters_Information.letters_info_logic import show_help_window
    from GUI.gui_utils import go_camera_letters_window

    # Configurar la lógica del botón "Regresar" (detiene la animación y vuelve al diccionario del alfabeto)
    button_go_back.config(command=lambda: (stop_letter_gif_animation(window), go_dictionary_alphabet_window(window)))

    # Configurar la lógica del botón "Inicio" (detiene la animación y vuelve a la ventana principal)
    button_go_home.config(command=lambda: (stop_letter_gif_animation(window), go_home_window(window)))

    # Configurar la lógica del botón "Ayuda" (muestra una ventana emergente con información de ayuda)
    button_help.config(command=lambda: show_help_window())

    # Configurar la lógica del botón "Cámara" (detiene la animación y abre la ventana de cámara para la letra actual)
    button_go_camera.config(command=lambda: (stop_letter_gif_animation(window),
                                             go_camera_letters_window(window, actual_letter)))

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes
    return button_go_back, button_go_home, button_go_camera, images
