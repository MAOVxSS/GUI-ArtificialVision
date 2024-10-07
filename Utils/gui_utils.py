# ARCHIVO PARA ESTILOS Y FUNCIONES
from tkinter import Canvas, Tk

# Configuración de las ventanas principales
WINDOW_CONFIG = {
    "bg": "#369FD6",  # Puedes cambiar esto si los colores difieren entre las ventanas
    "height": 768,
    "width": 1366,
    "bd": 0,
    "highlightthickness": 0,
    "relief": "ridge"
}


# Función para limpiar la ventana y configurarla
def setup_window(window, background_color="#369FD6"):
    """Configura la ventana con los estilos comunes y un color de fondo específico."""
    # Limpiar la ventana actual
    for widget in window.winfo_children():
        widget.destroy()

    # Configurar la ventana
    window.configure(bg=background_color)


# Función para crear un canvas con estilos comunes
def create_common_canvas(window, **kwargs):
    config = WINDOW_CONFIG.copy()
    config.update(kwargs)  # Actualizar con cualquier parámetro adicional específico

    canvas = Canvas(window, **config)
    canvas.place(x=0, y=0)
    return canvas


# Estilo común de botones
BUTTON_COMMON_CONFIG = {
    "borderwidth": 0,
    "highlightthickness": 0,
    "relief": "flat"
}

# Función de botón de regreso a la página de inicio
def on_back_button_home_click(window):
    print("Regresando al Inicio")

    # Importaciones de Home
    from GUI.Home.home_ui import create_home_window
    from GUI.Home.home_logic import setup_home_window_logic

    # Destruir la ventana actual (Information)
    window.destroy()

    # Crear una nueva ventana para Home
    new_window = Tk()
    new_window.geometry("1366x768")

    # Crear la interfaz "home"
    button_1, button_2, button_3, images = create_home_window(new_window)

    # Configurar la lógica de la ventana "home"
    setup_home_window_logic(new_window, button_1, button_2, button_3)

    # Iniciar el loop principal para la nueva ventana
    new_window.mainloop()



