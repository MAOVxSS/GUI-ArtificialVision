from pathlib import Path
from tkinter import Button, PhotoImage

# Funciones auxiliares
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Rutas
from Utils.paths import assets_letters_information_path


# Ruta hacia el archivo JSON e imágenes de la seña
def relative_to_assets_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)

def create_letters_information_window(window, actual_letter):
    # Se guarda la letra actual para mantenerla en las ventanas
    actual_letter = actual_letter

    # Configurar la ventana
    setup_window(window, background_color="#369FD6")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    canvas.create_rectangle(
        # Este rectángulo contendrá imagen extraída con la ruta del JSON "sign_path"
        609.0, 28.0, 1338.0, 529.0, fill="#233D4D", outline="")

    canvas.create_rectangle(
        # Este rectángulo contendrá el texto dinámico "description" del JSON data.json
        28.0, 219.0, 558.0, 738.0, fill="#233D4D", outline="")

    canvas.create_rectangle(
        # Este rectángulo es solo decorativo para el texto
        44.0, 43.0, 320.0, 191.0, fill="#F6AA1C", outline="")

    # Botones
    button_image_help = PhotoImage(file=relative_to_assets_data("pregunta.png"))
    images["help_icon"] = button_image_help
    button_help = Button(image=button_image_help, **BUTTON_COMMON_CONFIG)
    button_help.place(x=437.0, y=630.0, width=87.0, height=70.0)

    button_image_go_back = PhotoImage(file=relative_to_assets_data("button_1.png"))
    images["button_1"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=598.0, y=567.0, width=154.0, height=144.0)

    button_image_go_home = PhotoImage(file=relative_to_assets_data("button_2.png"))
    images["button_2"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=757.0, y=565.0, width=203.0, height=146.0)

    button_image_go_camera = PhotoImage(file=relative_to_assets_data("button_3.png"))
    images["button_3"] = button_image_go_camera
    button_go_camera = Button(image=button_image_go_camera, **BUTTON_COMMON_CONFIG)
    button_go_camera.place(x=963.0, y=565.0, width=378.0, height=146.0)

    # Lógica del botón "Regresar"
    from Utils.gui_utils import go_dictionary_alphabet_window
    from GUI.Letters_Information.letters_info_logic import stop_letter_gif_animation
    button_go_back.config(command=lambda: (stop_letter_gif_animation(window), go_dictionary_alphabet_window(window)))

    # Lógica del botón "Inicio"
    from Utils.gui_utils import go_home_window
    button_go_home.config(command=lambda: (stop_letter_gif_animation(window), go_home_window(window)))

    # Lógica del botón "Ayuda"
    from GUI.Letters_Information.letters_info_logic import show_help_window
    button_help.config(command=lambda: show_help_window())

    # Lógica del botón "Camera"
    from GUI.Camera.camera_logic import go_camera_window
    button_go_camera.config(command=lambda: (stop_letter_gif_animation(window),
                                             go_camera_window(window, actual_letter)))

    window.resizable(False, False)

    return button_go_back, button_go_home, button_go_camera, images
