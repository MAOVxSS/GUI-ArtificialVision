from tkinter import Tk, PhotoImage, Toplevel, Label

# Rutas a los archivos JSON con información de las letras y tips
from Utils.paths import assets_json_letters_info_path, assets_json_info_signs_path


import json

# Función para actualizar el icono de la letra elegida en la interfaz
def update_icon_letter(window, icon_path):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana-
    # Actualizar el icono dinámicamente
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(260.0, 290.0, image=image_icon)
    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_icon = image_icon


def show_tip_window(actual_letter):
    # Cargar los archivos JSON con información de las letras y los tips
    with open(assets_json_letters_info_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open(assets_json_info_signs_path, "r", encoding="utf-8") as tip_file:
        tip_data = json.load(tip_file)

    # Verificar si la letra actual existe en el archivo JSON
    if actual_letter in data:
        letter_data = data[actual_letter]

        # Crear la ventana emergente
        tip_window = Toplevel()
        tip_window.title(f"Tip para la letra {actual_letter}")

        # Definir el tamaño de la ventana emergente
        window_width = 573
        window_height = 430
        screen_width = tip_window.winfo_screenwidth()
        screen_height = tip_window.winfo_screenheight()

        # Definir el tamaño y la posición de la ventana emergente
        position_right = int(screen_width / 2 - window_width / 2)
        position_down = int(screen_height / 2 - window_height / 2)
        tip_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

        # Determinar si la letra es dinámica o estática
        movement_type = letter_data["movement"]

        # Obtener la descripción del tip según el tipo de movimiento
        if movement_type == "Estatico":
            description = tip_data["Static"]["Information"]
        elif movement_type == "Dinamico":
            description = tip_data["Dynamic"]["Information"]
        else:
            description = "No hay información disponible para este tipo de seña."

        # Convertir la descripción en texto si es una lista
        if isinstance(description, list):
            description_text = "\n".join(description)
        else:
            description_text = description

        # Crear una etiqueta con la descripción estilizada
        label_description = Label(tip_window, text=description_text, wraplength=window_width - 20, padx=10, pady=10,
                                  anchor="nw", justify="left", font=("Comic Sans MS", 16), fg="black")
        label_description.pack(expand=True, fill='both')

        # Mantener la referencia de la ventana para evitar que se elimine
        tip_window.label_description = label_description
