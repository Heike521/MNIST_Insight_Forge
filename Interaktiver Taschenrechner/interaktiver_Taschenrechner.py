import customtkinter as ctk
from Interface import CalculatorApp
from CalculatorFunction import Functions  
from ColorChoice import ColorChoiceWindow
# Starte CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
def run_with_farbwahl():
    # Startfenster (leeres Dummy-Rootfenster)
    root = ctk.CTk()
    root.withdraw()  # nicht anzeigen
    # Standardpalette als Ausgangspunkt
    default_palette = {
        "standard": ["#E0E0E0", "#1f1f1f"],
        "academic": ["#A0C4FF", "black"],
        "function": ["#FFD6A5", "black"],
        "tax": ["#FFADAD", "black"],
        "analyze": ["#CAFFBF", "black"]
    }

    def run_pocket_calculator(final_palette):
        app = CalculatorApp(final_palette)
        app.mainloop()

    # Zeige das Vorschaltfenster
    ColorChoiceWindow(root, default_palette, callback=run_pocket_calculator)

    root.mainloop()
if __name__ == "__main__":
    run_with_farbwahl()

