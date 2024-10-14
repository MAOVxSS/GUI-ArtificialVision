import json
from tkinter import Tk
from GUI.Letters_Information.letters_info_ui import create_letters_information_window
from Utils.gui_utils import center_window

# Ruta del archivo JSON
DATA_FILE_PATH = "GUI/Assets/Letters_Information/data.json"

# Función que se llama al hacer clic en un botón de una letra
def button_letter_click(window, letter):
    # Cargar el archivo JSON
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Verificar si la letra existe en el archivo JSON
    if letter in data:
        letter_data = data[letter]  # Obtener la información de la letra

        # Destruir la ventana actual (dictionary_alphabet_ui)
        window.destroy()

        # Crear una nueva ventana para la información de la letra
        new_window = Tk()
        center_window(new_window)

        # Crear la interfaz de la nueva ventana e insertar los datos dinámicos
        button_go_back, button_go_home, button_go_camera, images = create_letters_information_window(new_window)

        # Actualizar la ventana con la descripción y el tipo de la letra
        update_letter_info(new_window, letter_data["description"], letter_data["type"], letter_data["icon_path"],
                           letter_data["sign_path"])

        # Iniciar el loop de la nueva ventana
        new_window.mainloop()
    else:
        print(f"Letra '{letter}' no encontrada en el archivo JSON")

# Actualización de la función para incluir el ícono
def update_letter_info(window, description, type, icon_path, sign_path):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    # Si description es una lista, conviértela a una sola cadena de texto con saltos de línea
    if isinstance(description, list):
        description_text = "\n".join(description)
    else:
        description_text = description

    # Actualizar el texto de la descripción
    canvas.create_text(
        50.0, 240.0,  # Coordenadas
        anchor="nw",
        text=description_text,
        fill="white",
        font=("Comic Sans MS", 22 * -1),
        width=520  # Ajustar el ancho para que el texto haga wrap
    )

    # Actualizar el texto del tipo (letra)
    canvas.create_text(
        103.0, 78.0,  # Coordenadas
        anchor="nw",
        text=type,
        fill="#000000",
        font=("Comic Sans MS", 48 * -1)
    )

    # Actualizar el icono dinámicamente
    from tkinter import PhotoImage  # Asegurarse de que PhotoImage esté disponible
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(438.0, 117.0, image=image_icon)

    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_icon = image_icon

    # Mostrar la imagen de la seña
    image_sign = PhotoImage(file=sign_path)
    canvas.create_image(973.5, 278.5, image=image_sign)  # Colocar la imagen de la seña en el centro del rectángulo

    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_sign = image_sign
