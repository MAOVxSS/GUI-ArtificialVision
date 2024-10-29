import json
from tkinter import Tk, Toplevel, Label, PhotoImage
from PIL import ImageTk, Image
from GUI.Letters_Information.letters_info_ui import create_letters_information_window
from GUI.gui_utils import center_window
from pathlib import Path

# Ruta del archivo JSON que contiene la información sobre las letras
DATA_JSON_LETTERS_INFORMATION_PATH = "GUI/Assets/Letters_Information/letters_data.json"
from Utils.paths import assets_letters_information_path


# Función para generar la ruta completa hacia los archivos de recursos (JSON e imágenes)
def relative_to_assets_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)


# Variables globales para gestionar la animación del GIF en la ventana de información de letras
letter_gif_running = False  # Estado que indica si el GIF está siendo animado
letter_gif_animation_id = None  # id de la animación activa (para cancelar el ciclo `after`)


# Función para detener la animación del GIF en la ventana de información de letras
def stop_letter_gif_animation(window):
    global letter_gif_running, letter_gif_animation_id
    letter_gif_running = False  # Cambiar el estado para detener la animación
    if letter_gif_animation_id is not None:
        try:
            # Cancelar el ciclo `after` para detener la animación
            window.after_cancel(letter_gif_animation_id)
        except ValueError:
            pass  # En caso de error (por ejemplo, si ya fue cancelado), se ignora


# Función para mostrar el GIF animado en la ventana de información de letras
def show_letter_gif(window, gif_path, canvas):
    global letter_gif_running, letter_gif_animation_id
    letter_gif_running = False  # Detener cualquier animación de GIF anterior

    # Cancelar cualquier ciclo `after` activo para evitar conflictos
    if letter_gif_animation_id is not None:
        try:
            window.after_cancel(letter_gif_animation_id)
        except ValueError:
            pass

    # Cargar el GIF y dividirlo en frames
    gif_image = Image.open(gif_path)
    frames = []
    try:
        # Extraer todos los frames del GIF y almacenarlos en una lista
        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            frames.append(frame)
            gif_image.seek(len(frames))  # Avanzar al siguiente frame
    except EOFError:
        pass  # Terminar cuando no haya más frames

    # Función interna para actualizar el GIF en el canvas
    def update_gif(index):
        global letter_gif_animation_id
        if letter_gif_running:  # Continuar solo si la animación debe seguir corriendo
            canvas.itemconfig(gif_item, image=frames[index])
            # Programar la actualización del siguiente frame después de 100 ms
            letter_gif_animation_id = window.after(100, update_gif, (index + 1) % len(frames))

    # Crear la imagen en el canvas, ubicada en la posición especificada
    gif_item = canvas.create_image(973.5, 278.5, image=frames[0])  # Posición dentro del rectángulo
    letter_gif_running = True  # Iniciar la animación del GIF
    update_gif(0)  # Iniciar la actualización de los frames


# Función que se llama al hacer clic en un botón de una letra
def button_letter_click(window, letter):
    # Cargar el archivo JSON que contiene la información de las letras
    with open(DATA_JSON_LETTERS_INFORMATION_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Verificar si la letra existe en el archivo JSON
    if letter in data:
        letter_data = data[letter]  # Obtener la información de la letra

        # Destruir la ventana actual (cerrar la ventana `dictionary_alphabet_ui`)
        window.destroy()

        # Crear una nueva ventana para mostrar la información de la letra seleccionada
        new_window = Tk()
        center_window(new_window)  # Centrar la nueva ventana en la pantalla

        # Crear la interfaz de la nueva ventana e insertar los datos dinámicos
        button_go_back, button_go_home, button_go_camera, images = create_letters_information_window(
            new_window, letter_data["letter"])

        # Actualizar la nueva ventana con la descripción y el tipo de la letra
        update_letter_info(new_window, letter_data["description"], letter_data["type"],
                           letter_data["icon_path"], letter_data["sign_path"])

        # Iniciar el bucle de la nueva ventana
        new_window.mainloop()
    else:
        # Imprimir un mensaje de error si la letra no se encuentra en el archivo JSON
        print(f"Letra '{letter}' no encontrada en el archivo JSON")


# Función para actualizar los elementos de la información de la letra
def update_letter_info(window, description, type, icon_path, sign_path):
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
    canvas.create_text(103.0, 78.0, anchor="nw", text=type, fill="#000000",
                       font=("Comic Sans MS", 54 * -1))

    # Actualizar el icono de la letra
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(438.0, 117.0, image=image_icon)

    # Mantener una referencia de la imagen para evitar que sea recolectada por el recolector de basura
    window.image_icon = image_icon

    # Mostrar la imagen o GIF de la seña, dependiendo del formato del archivo
    if sign_path.endswith(".gif"):
        show_letter_gif(window, sign_path, canvas)
    else:
        # Mostrar la imagen de la seña (formato PNG)
        image_sign = PhotoImage(file=sign_path)
        canvas.create_image(973.5, 278.5, image=image_sign)  # Colocar la imagen en el rectángulo correspondiente
        window.image_sign = image_sign  # Mantener la referencia de la imagen


# Función para mostrar una ventana emergente con la imagen de los nombres de los dedos de la mano
def show_help_window():
    help_window = Toplevel()  # Crear una nueva ventana (emergente)

    # Tamaño de la ventana emergente
    window_width = 500
    window_height = 500

    # Obtener el tamaño de la pantalla para calcular la posición centrada
    screen_width = help_window.winfo_screenwidth()
    screen_height = help_window.winfo_screenheight()
    x_position = (screen_width // 2) - (window_width // 2)
    y_position = (screen_height // 2) - (window_height // 2)

    # Configurar la geometría de la ventana emergente
    help_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    help_window.title("Los dedos de la mano")

    # Cargar la imagen de los nombres de los dedos de la mano
    fingers_image = PhotoImage(file=relative_to_assets_data("dedos_de_la_mano.png"))

    # Crear una etiqueta para mostrar la imagen dentro de la ventana emergente
    label = Label(help_window, image=fingers_image)
    label.pack(padx=10, pady=10)

    # Mantener la referencia de la imagen para evitar que se elimine
    help_window.help_image = fingers_image
