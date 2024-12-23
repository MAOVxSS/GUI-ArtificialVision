from pathlib import Path
from tkinter import Button, PhotoImage, Label

# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Importación de recursos gráficos
from GUI.Camera.Camera_Letters.camera_letters_ui import relative_to_assets_camera
from GUI.Camera.Camera_Phrases.camera_phrases_model_logic import evaluate_phrases_model



# Función para crear la ventana de la cámara
def create_camera_phrases_window(window, actual_phrase):

    # Se guarda la letra actual para mantenerla en las ventanas
    actual_phrase = actual_phrase

    # Configurar la ventana principal con un color de fondo
    setup_window(window, background_color="#369FD6")

    # Crear un canvas común que actuará como contenedor de los elementos de la interfaz
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes para evitar que sean recolectadas por el recolector de basura
    images = {}

    # Crear un rectángulo que contendrá el video de la cámara web
    canvas.create_rectangle(533.0, 41.0, 1317.0, 617.0, fill="#233D4D", outline="")

    # Crear un `Label` para mostrar los frames del video
    video_label = Label(window)
    video_label.place(x=533.0, y=41.0, width=784.0, height=576.0)  # Posición y tamaño del rectángulo

    # Se manda a llamar a la función para cargar el video y el model de IA
    evaluate_phrases_model(video_label, window, actual_phrase)

    # Crear los botones de la interfaz
    # Botón para mostrar un consejo o tip
    button_image_tip = PhotoImage(file=relative_to_assets_camera("consejo.png"))
    images["consejo"] = button_image_tip
    button_tip = Button(image=button_image_tip, **BUTTON_COMMON_CONFIG)
    button_tip.place(x=68.0, y=425.0, width=384.0, height=127.0)

    # Botón para regresar a la ventana anterior "Información de Letras"
    button_image_go_back = PhotoImage(file=relative_to_assets_camera("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=60.0, y=574.0, width=160.0, height=139.0)

    # Botón para ir a la ventana principal "Inicio"
    button_image_go_home = PhotoImage(file=relative_to_assets_camera("inicio.png"))
    images["inicio"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=270.0, y=566.0, width=205.0, height=156.0)

    # Crear y colocar imágenes decorativas en el canvas
    image_image_1 = PhotoImage(file=relative_to_assets_camera("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(260.0, 123.0, image=image_image_1)

    image_image_5 = PhotoImage(file=relative_to_assets_camera("image_5.png"))
    images["image_5"] = image_image_5
    canvas.create_image(378.0, 234.0, image=image_image_5)

    # Importar funciones necesarias para la navegación entre ventanas
    from GUI.Phrases_Info.phrases_info_logic import button_phrase_click
    from GUI.gui_utils import go_home_window
    from GUI.Camera.Camera_Phrases.camera_phrases_logic import show_tip_phrases_window
    # from GUI.Camera.Camera_Letters.camera_letters_logic import show_tip_window
    from GUI.Camera.Camera_Letters.camera_letters_model_logic import stop_video_stream

    # Lógica del botón "Regresar"
    button_go_back.config(command=lambda: [stop_video_stream(video_label), button_phrase_click(window, actual_phrase)])

    # Logica del botón "Inicio"
    button_go_home.config(command=lambda: [stop_video_stream(video_label), go_home_window(window)])

    # Logica del botón "Tip"
    button_tip.config(command=lambda: show_tip_phrases_window(actual_phrase))

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes
    return button_tip, button_go_back, button_go_home, images
