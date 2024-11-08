from tkinter import Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_lessons_alphabet_path
from GUI.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG

def relative_to_assets(path: str) -> Path:
    return assets_lessons_alphabet_path / Path(path)

def create_lessons_alphabet_window(window):
    # Configurar la ventana con los estilos comunes
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    button_image_home = PhotoImage(file=relative_to_assets("inicio.png"))
    images["inicio"] = button_image_home
    button_home = Button(image=button_image_home, **BUTTON_COMMON_CONFIG)
    button_home.place(x=958.0, y=548.0, width=211.0, height=155.0)

    button_image_go_back = PhotoImage(file=relative_to_assets("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=808.0, y=549.0, width=150.0, height=154.0)

    button_image_random_lesson = PhotoImage(file=relative_to_assets("leccion_azar.png"))
    images["leccion_azar"] = button_image_random_lesson
    button_random_lesson = Button(image=button_image_random_lesson, **BUTTON_COMMON_CONFIG)
    button_random_lesson.place(x=702.0, y=397.0, width=579.0, height=144.0)

    button_image_complete_lesson = PhotoImage(file=relative_to_assets("leccion_completa.png"))
    images["leccion_completa"] = button_image_complete_lesson
    button_complete_lesson = Button(image=button_image_complete_lesson, **BUTTON_COMMON_CONFIG)
    button_complete_lesson.place(x=694.0, y=240.0, width=567.0, height=150.0)

    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(979.0, 127.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(303.0, 384.0, image=image_image_2)

    # Importar funciones necesarias para la navegación entre ventanas
    from GUI.gui_utils import go_lessons_window, go_home_window, go_complete_lesson_alphabet

    # Lógica del botón "Regresar"
    button_go_back.config(command=lambda: go_lessons_window(window))

    # Lógica del botón "Inicio"
    button_home.config(command=lambda: go_home_window(window))

    # Lógica del botón "Lección completa"
    button_complete_lesson.config(command=lambda: go_complete_lesson_alphabet(window))

    window.resizable(False, False)

    return (button_image_home, button_image_go_back, button_image_random_lesson,
            button_image_complete_lesson, images)