import json
from tkinter import Tk, Toplevel, Label, PhotoImage
from PIL import ImageTk, Image
from GUI.Letters_Information.letters_info_ui import create_letters_information_window
from Utils.gui_utils import center_window
from pathlib import Path

# Ruta del archivo JSON
DATA_FILE_PATH = "GUI/Assets/Letters_Information/data.json"
from Utils.paths import assets_letters_information_path

# Ruta hacia el archivo JSON e imágenes de la seña
def relative_to_assets_data(path: str) -> Path:
    return assets_letters_information_path / Path(path)

# Variables globales para gestionar el GIF en la ventana de información de letras
letter_gif_running = False
letter_gif_animation_id = None


# Función para detener la animación del GIF en la ventana de letras
def stop_letter_gif_animation(window):
    global letter_gif_running, letter_gif_animation_id
    letter_gif_running = False  # Detener la animación del GIF
    if letter_gif_animation_id is not None:
        try:
            window.after_cancel(letter_gif_animation_id)
        except ValueError:
            pass


# Función para mostrar el GIF animado en la ventana de información de letras
def show_letter_gif(window, gif_path, canvas):
    global letter_gif_running, letter_gif_animation_id
    letter_gif_running = False  # Detenemos cualquier animación de GIF anterior

    # Cancelar el ciclo `after` si hay uno activo
    if letter_gif_animation_id is not None:
        try:
            window.after_cancel(letter_gif_animation_id)
        except ValueError:
            pass

    # Cargar el GIF y dividir en frames
    gif_image = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            frames.append(frame)
            gif_image.seek(len(frames))  # Avanzar al siguiente frame
    except EOFError:
        pass  # Termina cuando ya no hay más frames

    # Definir la animación del GIF
    def update_gif(index):
        global letter_gif_animation_id
        if letter_gif_running:  # Solo continuar si la animación debe seguir corriendo
            canvas.itemconfig(gif_item, image=frames[index])
            letter_gif_animation_id = window.after(100, update_gif, (index + 1) % len(frames))

    gif_item = canvas.create_image(973.5, 278.5, image=frames[0])  # Posición dentro del rectángulo
    letter_gif_running = True  # Iniciar la animación
    update_gif(0)


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
        button_go_back, button_go_home, button_go_camera, images = create_letters_information_window(new_window,
                                                                                                     letter_data[
                                                                                                         "letter"])

        # Actualizar la ventana con la descripción y el tipo de la letra
        update_letter_info(new_window, letter_data["description"], letter_data["type"], letter_data["icon_path"],
                           letter_data["sign_path"])

        # Iniciar el loop de la nueva ventana
        new_window.mainloop()
    else:
        print(f"Letra '{letter}' no encontrada en el archivo JSON")


# Función para actualizar los elementos de la información de la letra
def update_letter_info(window, description, type, icon_path, sign_path):
    canvas = window.children['!canvas']  # Obtén el canvas de la ventana

    # Si description es una lista, conviértela a una sola cadena de texto con saltos de línea
    if isinstance(description, list):
        description_text = "\n".join(description)
    else:
        description_text = description

    # Actualizar el texto de la descripción
    canvas.create_text(50.0, 240.0, anchor="nw", text=description_text, fill="white", font=("Comic Sans MS", 22 * -1),
        width=500
    )

    # Actualizar el texto del tipo (letra)
    canvas.create_text(103.0, 78.0, anchor="nw", text=type, fill="#000000", font=("Comic Sans MS", 54 * -1)
    )

    # Actualizar el icono dinámicamente
    image_icon = PhotoImage(file=icon_path)
    canvas.create_image(438.0, 117.0, image=image_icon)

    # Mantener la referencia de la imagen para evitar que se elimine
    window.image_icon = image_icon

    # Si el archivo es un GIF, mostrarlo animado
    if sign_path.endswith(".gif"):
        show_letter_gif(window, sign_path, canvas)
    else:
        # Mostrar la imagen de la seña (PNG)
        image_sign = PhotoImage(file=sign_path)
        canvas.create_image(973.5, 278.5, image=image_sign)  # Colocar la imagen de la seña en el centro del rectángulo
        window.image_sign = image_sign


# Función para mostrar la ventana emergente con la imagen de los nombres de los dedos
def show_help_window():
    help_window = Toplevel()  # Crear una nueva ventana
    # Tamaño de la ventana emergente
    window_width = 500
    window_height = 500

    # Obtener el tamaño de la pantalla
    screen_width = help_window.winfo_screenwidth()
    screen_height = help_window.winfo_screenheight()

    # Calcular la posición x e y para centrar la ventana
    x_position = (screen_width // 2) - (window_width // 2)
    y_position = (screen_height // 2) - (window_height // 2)

    # Configurar la geometría de la ventana emergente para que esté centrada
    help_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    help_window.title("Los dedos de la mano")

    # Cargar la imagen
    fingers_image = PhotoImage(file=relative_to_assets_data("dedos_de_la_mano.png"))

    # Crear una etiqueta para mostrar la imagen
    label = Label(help_window, image=fingers_image)
    label.pack(padx=10, pady=10)

    # Mantener referencia de la imagen para evitar que se elimine
    help_window.help_image = fingers_image
