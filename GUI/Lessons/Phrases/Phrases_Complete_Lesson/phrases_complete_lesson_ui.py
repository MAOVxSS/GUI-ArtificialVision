from pathlib import Path
from tkinter import Button, PhotoImage, Label

from GUI.Lessons.Phrases.Phrases_Complete_Lesson.phrases_complete_lesson_logic import PHRASES
# Importación de funciones auxiliares de la GUI
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
# Importación de recursos gráficos
from GUI.Camera.Camera_Letters.camera_letters_ui import relative_to_assets_camera


# Función para crear la ventana de la cámara
def create_complete_lesson_phrases_window(window):
    # Inicializar el índice de la frase actual
    current_phrase_index = 0

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

    # Crear un Label para el progreso de la lección en la parte inferior del video
    progress_label = Label(window, text="1/27", font=("Arial", 16), fg="black", bg="#369FD6")
    progress_label.place(x=533 + 784 - 60, y=41 + 576 - 30)  # Posicionar la etiqueta de progreso

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
    from GUI.gui_utils import go_home_window, go_phrases_lessons_window
    # from GUI.Camera.Camera_Letters.camera_letters_logic import show_tip_window
    from GUI.Camera.Camera_Letters.camera_letters_model_logic import stop_video_stream

    # Lógica del botón "Regresar"
    button_go_back.config(command=lambda: [stop_video_stream(video_label), go_phrases_lessons_window(window)])

    # Logica del botón "Inicio"
    button_go_home.config(command=lambda: [stop_video_stream(video_label), go_home_window(window)])

    # Logica del botón "Tip"
    # button_tip.config(command=lambda: show_tip_window(actual_phrase))

    from GUI.Lessons.Phrases.Phrases_Complete_Lesson.phrases_complete_lesson_logic import (load_phrase_data, PHRASES,
    start_phrases_complete_recognition_cycle)
    from GUI.Camera.Camera_Phrases.camera_phrases_logic import update_icon_phrase

    # Cargar la primera frase y su ícono para mostrar en la interfaz
    phrase_data = load_phrase_data(PHRASES[current_phrase_index])  # Obtener datos de la letra actual
    if phrase_data:
        update_icon_phrase(window, phrase_data["icon_path"])  # Actualizar el icono de la letra en la interfaz

    # Inicia el reconocimiento
    start_phrases_complete_recognition_cycle(window, video_label, progress_label, current_phrase_index)

    # Desactivar la opción de redimensionar la ventana (para mantener un diseño fijo)
    window.resizable(False, False)

    # Retornar los botones y las referencias a las imágenes
    return button_tip, button_go_back, button_go_home, images
