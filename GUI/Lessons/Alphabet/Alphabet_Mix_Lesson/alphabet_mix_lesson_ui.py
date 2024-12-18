# Importación de módulos necesarios para la interfaz gráfica
from tkinter import Button, PhotoImage, Label
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
from GUI.Lessons.Alphabet.Alphabet_Mix_Lesson.alphabet_mix_lesson_logic import (
    start_letters_mix_recognition_cycle, load_letter_data, MIXED_LETTERS, relative_to_assets_camera)
from GUI.Camera.Camera_Letters.camera_letters_logic import update_icon_letter

# Crear la ventana de la lección al azar del abecedario
def create_alphabet_mix_lesson_window(window):
    # Configuración inicial de la ventana y el índice de la letra actual
    current_letter_index = 0
    setup_window(window, background_color="#369FD6")
    canvas = create_common_canvas(window)
    images = {}

    # Crear un `Label` para mostrar los frames del video
    video_label = Label(window)
    video_label.place(x=533.0, y=41.0, width=784.0, height=576.0)

    # Crear el Label para el progreso en la parte inferior del video
    progress_label = Label(window, text="0/10", font=("Arial", 16), fg="black", bg="#369FD6")
    progress_label.place(x=533 + 784 - 60, y=41 + 576 - 30)  # Ajustar posición en la esquina inferior derecha del video

    # Crear los botones de la interfaz
    button_image_tip = PhotoImage(file=relative_to_assets_camera("consejo.png"))
    images["consejo"] = button_image_tip
    button_tip = Button(image=button_image_tip, **BUTTON_COMMON_CONFIG)
    button_tip.place(x=68.0, y=425.0, width=384.0, height=127.0)

    button_image_go_back = PhotoImage(file=relative_to_assets_camera("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=60.0, y=574.0, width=160.0, height=139.0)

    button_image_go_home = PhotoImage(file=relative_to_assets_camera("inicio.png"))
    images["inicio"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=270.0, y=566.0, width=205.0, height=156.0)

    image_image_1 = PhotoImage(file=relative_to_assets_camera("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(260.0, 123.0, image=image_image_1)

    image_image_5 = PhotoImage(file=relative_to_assets_camera("image_5.png"))
    images["image_5"] = image_image_5
    canvas.create_image(378.0, 234.0, image=image_image_5)

    # Importar funciones necesarias para la navegación entre ventanas
    from GUI.Camera.Camera_Letters.camera_letters_model_logic import stop_video_stream
    from GUI.gui_utils import go_home_window, go_lessons_alphabet_window

    # Lógica del botón "Regresar"
    button_go_back.config(command=lambda: [stop_video_stream(video_label), go_lessons_alphabet_window(window)])

    # Lógica del botón "Inicio"
    button_go_home.config(command=lambda: [stop_video_stream(video_label), go_home_window(window)])

    # Cargar la primera letra de las seleccionadas al azar y su ícono
    letter_data = load_letter_data(MIXED_LETTERS[current_letter_index])
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])

    # Iniciar el ciclo de reconocimiento para las letras al azar y pasar el `progress_label` para actualizar el progreso
    start_letters_mix_recognition_cycle(window, video_label, progress_label, current_letter_index)

    return button_tip, button_go_back, button_go_home, images
