from GUI.Home.home_ui import create_home_window
from GUI.Home.home_logic import on_button_1_click, on_button_2_click, on_button_3_click

if __name__ == "__main__":
    # Crear la ventana principal "home"
    window, button_1, button_2, button_3, images = create_home_window()

    # Asociar las funciones a los botones
    button_1.config(command=on_button_1_click)
    button_2.config(command=on_button_2_click)
    button_3.config(command=on_button_3_click)

    # Iniciar la aplicación
    window.mainloop()