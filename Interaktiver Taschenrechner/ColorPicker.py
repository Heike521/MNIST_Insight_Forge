from tkinter import colorchooser
def choose_fg_color():
    '''
    Öffnet einen Farbwähler zur Auswahl einer Vordergrundfarbe (Textfarbe).

    Rückgabe:
    str: hex-Farbcode (z.B. "#000000") oder None bei Abbruch
    '''
    paint = colorchooser.askcolor(title="Textfarbe wählen")
    return paint[1] if paint else None


def choose_bg_color():
    '''
    Öffnet einen Farbwähler zur Auswahl einer Hintergrundfarbe (Buttonfarbe).

    Rückgabe:
    str: hex-Farbcode (z.B. "#FFFFFF") oder None bei Abbruch
    '''
    paint = colorchooser.askcolor(title="Hintergrundfarbe wählen")
    return paint[1] if paint else None
