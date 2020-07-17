import ctypes


def mouse_drag(x, y):
    mouse_down()
    ctypes.windll.user32.SetCursorPos(x, y)
    mouse_up()


def mouse_move(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)


def mouse_click():
    mouse_down()
    mouse_up()


def mouse_down():
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0) # left button down


def mouse_up():
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0) # left button up
