from tkinter import Tk
from GUI.Home.home_ui import create_home_window
from GUI.gui_utils import center_window

if __name__ == "__main__":
    # Crear la ventana principal
    window = Tk()

    # Centrar la ventana en la pantalla
    center_window(window)

    # Crear la interfaz "home"
    button_dictionary, button_information, button_lessons, images = create_home_window(window)

    # Iniciar la aplicación
    window.mainloop()