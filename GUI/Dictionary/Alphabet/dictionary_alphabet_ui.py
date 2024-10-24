from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from pathlib import Path

# Rutas
from Utils.paths import assets_dictionary_alphabet_path
from Utils.gui_utils import setup_window, create_common_canvas, BUTTON_COMMON_CONFIG


def relative_to_assets(path: str) -> Path:
    return assets_dictionary_alphabet_path / Path(path)


def create_alphabet_window(window):
    # Configurar la ventana con los estilos comunes
    setup_window(window, background_color="#379FD7")

    # Crear el canvas común
    canvas = create_common_canvas(window)

    # Mantener referencias a las imágenes
    images = {}

    # Botones
    button_image_a = PhotoImage(file=relative_to_assets("letter_a.png"))
    images["letter_a"] = button_image_a
    button_a = Button(image=button_image_a, **BUTTON_COMMON_CONFIG)
    button_a.place(x=172.0, y=62.0, width=127.0, height=114.0)

    button_image_b = PhotoImage(file=relative_to_assets("letter_b.png"))
    images["letter_b"] = button_image_b
    button_b = Button(image=button_image_b, **BUTTON_COMMON_CONFIG)
    button_b.place(x=324.0, y=62.0, width=127.0, height=114.0)

    button_image_c = PhotoImage(file=relative_to_assets("letter_c.png"))
    images["letter_c"] = button_image_c
    button_c = Button(image=button_image_c, **BUTTON_COMMON_CONFIG)
    button_c.place(x=471.0, y=62.0, width=127.0, height=114.0)

    button_image_d = PhotoImage(file=relative_to_assets("letter_d.png"))
    images["letter_d"] = button_image_d
    button_d = Button(image=button_image_d, **BUTTON_COMMON_CONFIG)
    button_d.place(x=620.0, y=62.0, width=127.0, height=114.0)

    button_image_e = PhotoImage(file=relative_to_assets("letter_e.png"))
    images["letter_e"] = button_image_e
    button_e = Button(image=button_image_e, **BUTTON_COMMON_CONFIG)
    button_e.place(x=769.0, y=62.0, width=127.0, height=114.0)

    button_image_f = PhotoImage(file=relative_to_assets("letter_f.png"))
    images["letter_f"] = button_image_f
    button_f = Button(image=button_image_f, **BUTTON_COMMON_CONFIG)
    button_f.place(x=918.0, y=62.0, width=127.0, height=114.0)

    button_image_g = PhotoImage(file=relative_to_assets("letter_g.png"))
    images["letter_g"] = button_image_g
    button_g = Button(image=button_image_g, **BUTTON_COMMON_CONFIG)
    button_g.place(x=1067.0, y=62.0, width=127.0, height=114.0)

    button_image_h = PhotoImage(file=relative_to_assets("letter_h.png"))
    images["letter_h"] = button_image_h
    button_h = Button(image=button_image_h, **BUTTON_COMMON_CONFIG)
    button_h.place(x=172.0, y=190.0, width=127.0, height=114.0)

    button_image_i = PhotoImage(file=relative_to_assets("letter_i.png"))
    images["letter_i"] = button_image_i
    button_i = Button(image=button_image_i, **BUTTON_COMMON_CONFIG)
    button_i.place(x=322.0, y=190.0, width=127.0, height=114.0)

    button_image_j = PhotoImage(file=relative_to_assets("letter_j.png"))
    images["letter_j"] = button_image_j
    button_j = Button(image=button_image_j, **BUTTON_COMMON_CONFIG)
    button_j.place(x=471.0, y=190.0, width=127.0, height=114.0)

    button_image_k = PhotoImage(file=relative_to_assets("letter_k.png"))
    images["letter_k"] = button_image_k
    button_k = Button(image=button_image_k, **BUTTON_COMMON_CONFIG)
    button_k.place(x=620.0, y=190.0, width=127.0, height=114.0)

    button_image_l = PhotoImage(file=relative_to_assets("letter_l.png"))
    images["letter_l"] = button_image_l
    button_l = Button(image=button_image_l, **BUTTON_COMMON_CONFIG)
    button_l.place(x=769.0, y=190.0, width=127.0, height=114.0)

    button_image_m = PhotoImage(file=relative_to_assets("letter_m.png"))
    images["letter_m"] = button_image_m
    button_m = Button(image=button_image_m, **BUTTON_COMMON_CONFIG)
    button_m.place(x=918.0, y=190.0, width=127.0, height=114.0)

    button_image_n = PhotoImage(file=relative_to_assets("letter_n.png"))
    images["letter_n"] = button_image_n
    button_n = Button(image=button_image_n, **BUTTON_COMMON_CONFIG)
    button_n.place(x=1067.0, y=190.0, width=127.0, height=114.0)

    button_image_nn = PhotoImage(file=relative_to_assets("letter_nn.png"))
    images["letter_nn"] = button_image_nn
    button_nn = Button(image=button_image_nn, **BUTTON_COMMON_CONFIG)
    button_nn.place(x=172.0, y=318.0, width=127.0, height=114.0)

    button_image_o = PhotoImage(file=relative_to_assets("letter_o.png"))
    images["letter_o"] = button_image_o
    button_o = Button(image=button_image_o, **BUTTON_COMMON_CONFIG)
    button_o.place(x=322.0, y=318.0, width=127.0, height=114.0)

    button_image_p = PhotoImage(file=relative_to_assets("letter_p.png"))
    images["letter_p"] = button_image_p
    button_p = Button(image=button_image_p, **BUTTON_COMMON_CONFIG)
    button_p.place(x=471.0, y=318.0, width=127.0, height=114.0)

    button_image_q = PhotoImage(file=relative_to_assets("letter_q.png"))
    images["letter_q"] = button_image_q
    button_q = Button(image=button_image_q, **BUTTON_COMMON_CONFIG)
    button_q.place(x=620.0, y=318.0, width=127.0, height=114.0)

    button_image_r = PhotoImage(file=relative_to_assets("letter_r.png"))
    images["letter_r"] = button_image_r
    button_r = Button(image=button_image_r, **BUTTON_COMMON_CONFIG)
    button_r.place(x=769.0, y=318.0, width=127.0, height=114.0)

    button_image_s = PhotoImage(file=relative_to_assets("letter_s.png"))
    images["letter_s"] = button_image_s
    button_s = Button(image=button_image_s, **BUTTON_COMMON_CONFIG)
    button_s.place(x=918.0, y=318.0, width=127.0, height=114.0)

    button_image_t = PhotoImage(file=relative_to_assets("letter_t.png"))
    images["letter_t"] = button_image_t
    button_t = Button(image=button_image_t, **BUTTON_COMMON_CONFIG)
    button_t.place(x=1067.0, y=318.0, width=127.0, height=114.0)

    button_image_u = PhotoImage(file=relative_to_assets("letter_u.png"))
    images["letter_u"] = button_image_u
    button_u = Button(image=button_image_u, **BUTTON_COMMON_CONFIG)
    button_u.place(x=245.0, y=455.0, width=127.0, height=114.0)

    button_image_v = PhotoImage(file=relative_to_assets("letter_v.png"))
    images["letter_v"] = button_image_v
    button_v = Button(image=button_image_v, **BUTTON_COMMON_CONFIG)
    button_v.place(x=394.0, y=455.0, width=127.0, height=114.0)

    button_image_z = PhotoImage(file=relative_to_assets("letter_z.png"))
    images["letter_z"] = button_image_z
    button_z = Button(image=button_image_z, **BUTTON_COMMON_CONFIG)
    button_z.place(x=990.0, y=455.0, width=127.0, height=114.0)

    button_image_y = PhotoImage(file=relative_to_assets("letter_y.png"))
    images["letter_y"] = button_image_y
    button_y = Button(image=button_image_y, **BUTTON_COMMON_CONFIG)
    button_y.place(x=841.0, y=455.0, width=127.0, height=114.0)

    button_image_x = PhotoImage(file=relative_to_assets("letter_x.png"))
    images["letter_x"] = button_image_x
    button_x = Button(image=button_image_x, **BUTTON_COMMON_CONFIG)
    button_x.place(x=692.0, y=455.0, width=127.0, height=114.0)

    button_image_w = PhotoImage(file=relative_to_assets("letter_w.png"))
    images["letter_w"] = button_image_w
    button_w = Button(image=button_image_w, **BUTTON_COMMON_CONFIG)
    button_w.place(x=543.0, y=455.0, width=127.0, height=114.0)

    button_image_go_back = PhotoImage(file=relative_to_assets("regresar.png"))
    images["regresar"] = button_image_go_back
    button_go_back = Button(image=button_image_go_back, **BUTTON_COMMON_CONFIG)
    button_go_back.place(x=235.0, y=588.0, width=491.0, height=146.0)

    button_image_go_home = PhotoImage(file=relative_to_assets("inicio.png"))
    images["inicio"] = button_image_go_home
    button_go_home = Button(image=button_image_go_home, **BUTTON_COMMON_CONFIG)
    button_go_home.place(x=733.0, y=588.0, width=408.0, height=146.0)

    # Lógica del botón "Regresar"
    from Utils.gui_utils import go_dictionary_window
    button_go_back.config(command=lambda: go_dictionary_window(window))

    # Lógica del botón "Inicio"
    from Utils.gui_utils import go_home_window
    button_go_home.config(command=lambda: go_home_window(window))

    # Lógica de los botones para la información de la letrás
    from GUI.Letters_Information.letters_info_logic import button_letter_click

    button_a.config(command=lambda: button_letter_click(window, "A"))
    button_b.config(command=lambda: button_letter_click(window, "B"))
    button_c.config(command=lambda: button_letter_click(window, "C"))
    button_d.config(command=lambda: button_letter_click(window, "D"))
    button_e.config(command=lambda: button_letter_click(window, "E"))
    button_f.config(command=lambda: button_letter_click(window, "F"))
    button_g.config(command=lambda: button_letter_click(window, "G"))
    button_h.config(command=lambda: button_letter_click(window, "H"))
    button_i.config(command=lambda: button_letter_click(window, "I"))
    button_j.config(command=lambda: button_letter_click(window, "J"))
    button_k.config(command=lambda: button_letter_click(window, "K"))
    button_l.config(command=lambda: button_letter_click(window, "L"))
    button_m.config(command=lambda: button_letter_click(window, "M"))
    button_n.config(command=lambda: button_letter_click(window, "N"))
    button_nn.config(command=lambda: button_letter_click(window, "NN"))
    button_o.config(command=lambda: button_letter_click(window, "O"))
    button_p.config(command=lambda: button_letter_click(window, "P"))
    button_q.config(command=lambda: button_letter_click(window, "Q"))
    button_r.config(command=lambda: button_letter_click(window, "R"))
    button_s.config(command=lambda: button_letter_click(window, "S"))
    button_t.config(command=lambda: button_letter_click(window, "T"))
    button_u.config(command=lambda: button_letter_click(window, "U"))
    button_v.config(command=lambda: button_letter_click(window, "V"))
    button_w.config(command=lambda: button_letter_click(window, "W"))
    button_x.config(command=lambda: button_letter_click(window, "X"))
    button_y.config(command=lambda: button_letter_click(window, "Y"))
    button_z.config(command=lambda: button_letter_click(window, "Z"))


    window.resizable(False, False)

    return (button_a, button_b, button_c, button_d, button_e, button_f, button_g, button_h,
            button_i, button_j, button_k, button_l, button_m, button_n, button_nn, button_o, button_p,
            button_q, button_r, button_s, button_t, button_u, button_v, button_w, button_x,
            button_y, button_z, button_go_back, button_go_home, images)
