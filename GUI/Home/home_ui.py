from pathlib import Path
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from PIL import Image, ImageTk

# Rutas
from Utils.paths import assets_path


def relative_to_assets(path: str) -> Path:
    return assets_path / Path(path)


def create_home_window():
    window = Tk()
    window.geometry("1366x768")
    window.configure(bg="#369FD6")

    canvas = Canvas(
        window,
        bg="#369FD6",
        height=768,
        width=1366,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.place(x=0, y=0)

    # Mantener referencias a las imágenes
    images = {}

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    images["button_1"] = button_image_1
    button_1 = Button(image=button_image_1, borderwidth=0, highlightthickness=0, relief="flat")
    button_1.place(x=88.0, y=478.0, width=446.0, height=130.0)

    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
    images["button_2"] = button_image_2
    button_2 = Button(image=button_image_2, borderwidth=0, highlightthickness=0, relief="flat")
    button_2.place(x=858.0, y=486.0, width=446.0, height=130.0)

    button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
    images["button_3"] = button_image_3
    button_3 = Button(image=button_image_3, borderwidth=0, highlightthickness=0, relief="flat")
    button_3.place(x=1224.0, y=629.0, width=115.0, height=121.0)

    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    images["image_1"] = image_image_1
    canvas.create_image(683.0, 270.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    images["image_2"] = image_image_2
    canvas.create_image(239.0, 97.0, image=image_image_2)

    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
    images["image_3"] = image_image_3
    canvas.create_image(1052.0, 99.0, image=image_image_3)

    image_image_4 = PhotoImage(file=relative_to_assets("image_4.png"))
    images["image_4"] = image_image_4
    canvas.create_image(1030.0, 704.0, image=image_image_4)

    # Cargar y mostrar el GIF en lugar del rectángulo
    gif_path = relative_to_assets("logo-tese-animado-2.gif")
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
        canvas.itemconfig(gif_item, image=frames[index])
        window.after(100, update_gif, (index + 1) % len(frames))

    gif_item = canvas.create_image(647.0, 104.5, image=frames[0])
    update_gif(0)

    window.resizable(False, False)

    return window, button_1, button_2, button_3, images
