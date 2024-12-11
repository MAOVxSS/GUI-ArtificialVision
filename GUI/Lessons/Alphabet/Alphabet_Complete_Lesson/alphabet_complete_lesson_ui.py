from tkinter import Button, PhotoImage, Label
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG
from GUI.Lessons.Alphabet.Alphabet_Complete_Lesson.alphabet_complete_lesson_logic import (
    start_letters_complete_recognition_cycle, load_letter_data, ALPHABET, relative_to_assets_camera)
from GUI.Camera.Camera_Letters.camera_letters_model_logic import stop_video_stream
from GUI.gui_utils import go_home_window, go_lessons_alphabet_window
from GUI.Camera.Camera_Letters.camera_letters_logic import update_icon_letter


def create_complete_lesson_alphabet_window(window):
    """
    Crea y configura la ventana principal de la lección completa del abecedario en lenguaje de señas.

    Argumentos:
    window -- ventana principal de la aplicación sobre la cual se creará la lección.
    """
    # Inicializar el índice de la letra actual en 0 (corresponde a 'A' en el alfabeto)
    current_letter_index = 0

    # Configurar la ventana principal con un color de fondo específico y un canvas común para elementos gráficos
    setup_window(window, background_color="#369FD6")
    canvas = create_common_canvas(window)  # Crear un lienzo para elementos visuales comunes
    images = {}  # Diccionario para almacenar imágenes utilizadas en la interfaz

    # Crear un Label para mostrar los frames del video capturado
    video_label = Label(window)
    video_label.place(x=533.0, y=41.0, width=784.0, height=576.0)  # Ubicación y tamaño del área de video

    # Crear un Label para el progreso de la lección en la parte inferior del video
    progress_label = Label(window, text="1/27", font=("Arial", 16), fg="black", bg="#369FD6")
    progress_label.place(x=533 + 784 - 60, y=41 + 576 - 30)  # Posicionar la etiqueta de progreso

    # Crear botones de la interfaz gráfica y asignar imágenes correspondientes
    button_image_tip = PhotoImage(file=relative_to_assets_camera("consejo.png"))  # Cargar imagen de "Consejo"
    images["consejo"] = button_image_tip  # Guardar referencia de la imagen en el diccionario
    button_tip = Button(image=button_image_tip, **BUTTON_COMMON_CONFIG)  # Crear botón con configuración común
    button_tip.place(x=68.0, y=425.0, width=384.0, height=127.0)  # Posicionar botón de "Consejo"

    # Configuración del botón "Regresar"
    button_image_go_back = PhotoImage(file=relative_to_assets_camera("regresar.png"))  # Cargar imagen de "Regresar"
    images["regresar"] = button_image_go_back  # Guardar referencia de la imagen en el diccionario
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)  # Crear botón de "Regresar"
    button_go_back.place(x=60.0, y=574.0, width=160.0, height=139.0)  # Posicionar botón de "Regresar"

    # Configuración del botón "Inicio"
    button_image_go_home = PhotoImage(file=relative_to_assets_camera("inicio.png"))  # Cargar imagen de "Inicio"
    images["inicio"] = button_image_go_home  # Guardar referencia de la imagen en el diccionario
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)  # Crear botón de "Inicio"
    button_go_home.place(x=270.0, y=566.0, width=205.0, height=156.0)  # Posicionar botón de "Inicio"

    # Añadir imágenes decorativas en el canvas de la interfaz
    image_image_1 = PhotoImage(file=relative_to_assets_camera("image_1.png"))  # Cargar primera imagen decorativa
    images["image_1"] = image_image_1  # Guardar referencia de la imagen en el diccionario
    canvas.create_image(260.0, 123.0, image=image_image_1)  # Posicionar primera imagen en el canvas

    image_image_5 = PhotoImage(file=relative_to_assets_camera("image_5.png"))  # Cargar segunda imagen decorativa
    images["image_5"] = image_image_5  # Guardar referencia de la imagen en el diccionario
    canvas.create_image(378.0, 234.0, image=image_image_5)  # Posicionar segunda imagen en el canvas

    # Configuración de la acción del botón "Regresar" para detener el video y volver a la lección del alfabeto
    button_go_back.config(command=lambda: [stop_video_stream(video_label), go_lessons_alphabet_window(window)])

    # Configuración de la acción del botón "Inicio" para detener el video y regresar a la ventana principal
    button_go_home.config(command=lambda: [stop_video_stream(video_label), go_home_window(window)])

    # Cargar la primera letra ("A") y su ícono para mostrar en la interfaz
    letter_data = load_letter_data(ALPHABET[current_letter_index])  # Obtener datos de la letra actual
    if letter_data:
        update_icon_letter(window, letter_data["icon_path"])  # Actualizar el icono de la letra en la interfaz

    # Iniciar el ciclo de reconocimiento de señas, pasando la etiqueta de progreso para que se actualice dinámicamente
    start_letters_complete_recognition_cycle(window, video_label, progress_label, current_letter_index)

    # Retornar referencias a los botones y al diccionario de imágenes utilizadas
    return button_tip, button_go_back, button_go_home, images
