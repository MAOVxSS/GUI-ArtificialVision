from pathlib import Path
from tkinter import Button, PhotoImage, Label
import cv2

# Funciones auxiliares
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Rutas
from Utils.paths import assets_camera_path


def relative_to_assets(path: str) -> Path:
    return assets_camera_path / Path(path)


def create_camera_window(window, actual_letter):
    # Se guarda la letra actual para mantenerla en las ventanas
    actual_letter = actual_letter

    # Configurar la ventana
    setup_window(window, background_color="#369FD6")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Rectángulo para contener la el video de la camara web
    canvas.create_rectangle(533.0, 41.0, 1317.0, 617.0, fill="#233D4D", outline="")

    # Crear un Label para mostrar los frames del video
    video_label = Label(window)
    video_label.place(x=533.0, y=41.0, width=784.0, height=576.0)  # Posición y tamaño del rectángulo

    # Botones
    button_image_tip = PhotoImage(file=relative_to_assets("consejo.png"))
    images["consejo"] = button_image_tip
    button_tip = Button(image=button_image_tip, **BUTTON_COMMON_CONFIG)
    button_tip.place(x=294.0, y=370.0, width=169.0, height=169.0)

    button_image_go_back = PhotoImage(file=relative_to_assets("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=60.0, y=574.0, width=160.0, height=139.0
                         )

    button_image_go_home = PhotoImage(file=relative_to_assets("inicio.png"))
    images["inicio"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=270.0, y=566.0, width=205.0, height=156.0)

    # Imagenes
    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(260.0, 123.0, image=image_image_1)

    image_image_5 = PhotoImage(file=relative_to_assets("image_5.png"))
    images["image_5"] = image_image_5
    canvas.create_image(378.0, 234.0, image=image_image_5)

    image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
    images["image_4"] = image_image_4
    canvas.create_image(162.0, 464.0, image=image_image_4)

    # Esta imagen cambiara dinamicamente segun el resultado de la seña
    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(925.0, 668.5948486328125, image=image_image_2)

    # Logica para iniciar la camara y toma de video
    from GUI.Camera.camera_logic import start_video_stream, stop_video_stream
    start_video_stream(video_label)

    # Lógica del botón "Regresar"
    from GUI.Letters_Information.letters_info_logic import button_letter_click
    button_go_back.config(command=lambda: [stop_video_stream(video_label), button_letter_click(window, actual_letter)])

    # Logica del botón "Inicio"
    from Utils.gui_utils import go_home_window
    button_go_home.config(command=lambda: [stop_video_stream(video_label), go_home_window(window)])

    # Logica del botón "Tip"
    from GUI.Camera.camera_logic import show_tip_window
    button_tip.config(command=lambda: show_tip_window(actual_letter))

    window.resizable(False, False)

    return button_tip, button_go_back, button_go_home, images
