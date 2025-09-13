from tkinter import colorchooser
import customtkinter as ctk

class ColorChoiceWindow(ctk.CTkToplevel):
    '''
    Vorschaltfenster zur interaktiven Farbauswahl vor dem Start des Rechners.
        
        Parameter:
        - parent: Root-Fenster (meist versteckt)
        - color_palette: Dict mit allen Kategorien und Standardfarben
        - callback: Funktion, die aufgerufen wird, nachdem Start gedrückt wurde
    '''
    def __init__(self, parent, color_palette, callback):
        super().__init__(parent)
        self.title("Farbwahl vor Start")
        self.geometry("400x400")
        self.resizable(False, False)
        self.color_palette = color_palette
        self.callback = callback
        
        row = 0
        for kategorie in self.color_palette.keys():
            ctk.CTkLabel(self, text=f"{kategorie}").grid(row=row, column=0, 
                                                            padx=10, pady=5)
            ctk.CTkButton(self, text="Hintergrund wählen", 
                        command=lambda k=kategorie: self.pick_color(k, 0)).grid(row=row, 
                        column=1, padx=5)
            ctk.CTkButton(self, text="Textfarbe wählen", 
                        command=lambda k=kategorie: self.pick_color(k, 1)).grid(row=row, 
                        column=2, padx=5)
            row += 1

        self.start_button = ctk.CTkButton(self, text="Taschenrechner starten", 
                                        command=self.start_app)
        self.start_button.grid(row=row, column=0, columnspan=3, pady=20)
    def pick_color(self, kategorie, index):
        farbe = colorchooser.askcolor(title="Farbe wählen")[1]
        if farbe:
            self.color_palette[kategorie][index] = farbe

    def start_app(self):
        self.callback(self.color_palette)
        self.destroy()