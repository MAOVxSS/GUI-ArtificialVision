import json
from tkinter import Tk, Toplevel, Label, PhotoImage
from PIL import ImageTk, Image
from GUI.Phrases_Info.phrases_info_ui import create_phrases_information_window
from GUI.gui_utils import center_window
from pathlib import Path

# Ruta del archivo JSON que contiene la información sobre las letras
from Utils.paths import assets_json_phrases_info_path

from Utils.paths import assets_letters_information_path


# Función para generar la ruta completa hacia los archivos de recursos (JSON e imágenes)
def relative_to_assets_letters_information_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)


# Función que se llama al hacer clic en un botón de una frase
def button_phrase_click(window, phrase):
    # Cargar el archivo JSON que contiene la información de las letras
    with open(assets_json_phrases_info_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Verificar si la letra existe en el archivo JSON
    if phrase in data:
        phrase_data = data[phrase]  # Obtener la información de la letra

        # Destruir la ventana actual (cerrar la ventana `dictionary_alphabet_ui`)
        window.destroy()

        # Crear una nueva ventana para mostrar la información de la letra seleccionada
        new_window = Tk()
        center_window(new_window)  # Centrar la nueva ventana en la pantalla

        # Crear la interfaz de la nueva ventana e insertar los datos dinámicos
        button_go_back, button_go_home, button_go_camera, images = create_phrases_information_window(
            new_window, phrase_data["complete_phrase"])

        # Actualizar la nueva ventana con la descripción y el tipo de la letra
        update_phrase_info(new_window, phrase_data["description"], phrase_data["type"],
                           phrase_data["icon_path"], phrase_data["phrase_path"])

        # Iniciar el bucle de la nueva ventana
        new_window.mainloop()
    else:
        # Imprimir un mensaje de error si la letra no se encuentra en el archivo JSON
        print(f"Letra '{phrase}' no encontrada en el archivo JSON")


# Función para actualizar los elementos de la información de una frase
def update_phrase_info(window, description, type, icon_path, phrase_path):
    from GUI.Letters_Information.letters_info_logic import show_letter_gif
    canvas = window.children['!canvas']  # Obtener el canvas de la ventana

    # Si la descripción es una lista, convertirla a una cadena con saltos de línea
    if isinstance(description, list):
        description_text = "\n".join(description)
    else:
        description_text = description

    # Actualizar el texto de la descripción en el canvas
    canvas.create_text(50.0, 240.0, anchor="nw", text=description_text, fill="white",
                       font=("Comic Sans MS", 22 * -1), width=500)

    # Actualizar el texto del tipo de la letra
    canvas.create_text(50.0, 80.0, anchor="nw", text=type, fill="#000000",
                       font=("Comic Sans MS", 54 * -1))

    update_icon_phrase(window, icon_path)

    # Mostrar la imagen o GIF de la seña, dependiendo del formato del archivo
    if phrase_path.endswith(".gif"):
        show_letter_gif(window, phrase_path, canvas)
    else:
        # Mostrar la imagen de la seña (formato PNG)
        image_sign = PhotoImage(file=phrase_path)
        canvas.create_image(973.5, 278.5, image=image_sign)  # Colocar la imagen en el rectángulo correspondiente
        window.image_sign = image_sign  # Mantener la referencia de la imagen


# Función para actualizar el icono de la letra elegida en la interfaz
def update_icon_phrase(window, icon_path):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana-
    # Actualizar el icono dinámicamente
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(396.0, 118.0, image=image_icon)
    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_icon = image_icon
