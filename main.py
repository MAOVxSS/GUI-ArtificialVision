from tkinter import Tk
from GUI.Home.home_ui import create_home_window
from GUI.Home.home_logic import setup_home_window_logic

if __name__ == "__main__":
    # Crear la ventana principal
    window = Tk()
    window.geometry("1366x768")

    # Crear la interfaz "home"
    button_1, button_2, button_3, images = create_home_window(window)

    # Configurar la lógica de la ventana "home"
    setup_home_window_logic(window, button_1, button_2, button_3)

    # Iniciar la aplicación
    window.mainloop()