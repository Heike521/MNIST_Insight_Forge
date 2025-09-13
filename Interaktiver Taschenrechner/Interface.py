import customtkinter as ctk
from tkinter import END
from ColorPicker import choose_fg_color, choose_bg_color
from CalculatorFunction import Functions
class CalculatorApp(ctk.CTk):
    '''
    Hauptklasse für den interaktiven Taschenrechner.
    '''
    def __init__(self, color_palette):
        super().__init__()
        self.title("Taschenrechner mit Farbwahl")
        self.geometry("400x750")
        self.resizable(False, False)
        
        self.color_palette = color_palette  # ← übergebene Farbpalette

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.entry = ctk.CTkEntry(self, width=360, height=50, 
                                    font=("Arial", 20), justify="right")
        self.entry.grid(row=0, column=0, columnspan=5, padx=10, pady=20)

        self.functions = Functions(self.entry)

        self.buttons = [
            ("sin", 1, 0, "academic"), ("cos", 1, 1, "academic"), 
            ("tan", 1, 2, "academic"), ("log", 1, 3, "academic"), 
            ("ln", 1, 4, "academic"),
            
            ("1/x", 2, 0, "academic"), ("x²", 2, 1, "academic"),
            ("√", 2, 2, "academic"), ("!", 2, 3, "academic"),
            
            ("(", 3, 0, "function"), (")", 3, 1, "function"),
            ("%", 3, 2, "function"), ("C", 3, 3, "tax"),
            
            ("7", 4, 0, "standard"), ("8", 4, 1, "standard"), 
            ("9", 4, 2, "standard"), ("/", 4, 3, "function"), 
            
            ("4", 5, 0, "standard"), ("5", 5, 1, "standard"), 
            ("6", 5, 2, "standard"), ("*", 5, 3, "function"),
            
            ("1", 6, 0, "standard"), ("2", 6, 1, "standard"), 
            ("3", 6, 2, "standard"), ("-", 6, 3, "function"),
            
            ("0", 7, 0, "standard"), ("±", 7, 1, "function"),
            (".", 7, 2, "standard"), ("+", 7, 3, "function"),
            ("=", 7, 4, "analyze")
        ]

        self.button_widgets = []
        self.create_color_selector()
        self.create_buttons()

        for i in range(5):
            self.columnconfigure(i, weight=1)
        for i in range(8):
            self.rowconfigure(i, weight=1)

    def create_color_selector(self):
        '''
        Erstellt eine Auswahl für die Farbkategorie + zwei Buttons zum Ändern 
        von Hintergrund- und Textfarbe.
        '''
        self.color_choice = ctk.CTkOptionMenu(self, values=list(self.color_palette.keys()))
        self.color_choice.set("standard")
        self.color_choice.grid(row=8, column=0, columnspan=2, padx=5, pady=5, 
                                sticky="ew")

        self.bg_button = ctk.CTkButton(self, text="Hintergrundfarbe ändern", 
                                        command=self.change_background_color)
        self.bg_button.grid(row=8, column=2, columnspan=1, padx=5, pady=5, 
                            sticky="ew")

        self.fg_button = ctk.CTkButton(self, text="Textfarbe ändern", 
                                        command=self.change_text_color)
        self.fg_button.grid(row=8, column=3, columnspan=2, padx=5, pady=5, 
                            sticky="ew")
    def change_background_color(self):
        '''
        Öffnet ColorPicker für Hintergrundfarbe der gewählten Kategorie.
        '''
        kategorie = self.color_choice.get()
        new_paint = choose_bg_color()
        if new_paint:
            self.color_palette[kategorie][0] = new_paint
            self.update_button_colors(kategorie)

    def change_text_color(self):
        '''
        Öffnet ColorPicker für Textfarbe der gewählten Kategorie.
        '''
        kategorie = self.color_choice.get()
        new_paint = choose_fg_color()
        if new_paint:
            self.color_palette[kategorie][1] = new_paint
            self.update_button_colors(kategorie)

    def create_buttons(self):
        '''
        Erstellt alle Buttons anhand der Buttonliste.
        '''
        for (text, row, col, kategorie) in self.buttons:
            cmd = self.get_command(text)
            fg_color, text_color = self.color_palette[kategorie]
            btn = ctk.CTkButton(self, text=text, command=cmd, fg_color=fg_color,
                                text_color=text_color)
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.button_widgets.append((btn, kategorie))

    def update_button_colors(self, new_color):
        '''
        Aktualisiert die Farbe aller Buttons einer bestimmten Kategorie.
        '''
        if new_color not in self.color_palette:
            return
        fg_color, text_color = self.color_palette[new_color]
        for btn, kategorie in self.button_widgets:
            if kategorie == new_color:
                btn.configure(fg_color=fg_color, text_color=text_color)
    def get_command(self, text):
        f = self.functions
        return {
            "=": f.analyze,
            "C": f.delete,
            "±": f.key_signature,
            "√": f.root,
            "x²": f.potency,
            "sin": f.sinus,
            "cos": f.cosinus,
            "tan": f.tangens,
            "log": f.log10,
            "ln": f.ln,
            "1/x": f.one_through,
            "!": f.factorial
        }.get(text, lambda t=text: f.push(t))