from tkinter import END
import math
class Functions:
    '''
    Klasse zur Ausführung mathematischer Operationen auf einem GUI-Eingabefeld 
    (Entry).
    
    Diese Klasse kapselt alle Operationen, die auf dem eingegebenen Wert im 
    Entry-Feld durchgeführt werden sollen, z.B. Grundrechenarten, Wurzel, Potenz, 
    trigonometrische Funktionen, Logarithmen, Fakultät u.v.m.
    '''
    def __init__(self, entry):
        '''
        Konstruktor, speichert das übergebene Entry-Feld für alle Operationen.

        Parameter:
        entry (tk.Entry oder ctk.CTkEntry): Das Eingabefeld für die Taschen-
        rechnerfunktionen.
        '''
        self.entry = entry
    def push(self, z):
        '''
        Fügt das Zeichen `z` am Ende des Eingabefeldes ein.

        Parameter:
        z (str): Zeichen oder Zahl, die eingefügt werden soll.
        '''
        self.entry.insert(END, z)

    def delete(self):
        '''
        Löscht den gesamten Inhalt des Eingabefeldes.
        '''
        self.entry.delete(0, END)

    def analyze(self):
        '''
        Parst den Ausdruck im Eingabefeld und berechnet das Ergebnis.
        Prozentzeichen (%) wird in '/100' umgewandelt.
        '''
        try:
            printout = self.entry.get().replace('%', '/100')
            result = eval(printout)
            self.entry.delete(0, END)
            self.entry.insert(0, str(result))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def key_signature(self):
        '''
        Wechselt das Vorzeichen des aktuellen Wertes im Eingabefeld.
        '''
        try:
            worth = self.entry.get()
            if worth:
                if worth.startswith("-"):
                    self.entry.delete(0, 1)
                else:
                    self.entry.insert(0, "-")
        except:
            pass

    def root(self):
        '''
        Berechnet die Quadratwurzel des eingegebenen Wertes.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.sqrt(worth)))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def potency(self):
        '''
        Quadriert den eingegebenen Wert.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.pow(worth, 2)))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def sinus(self):
        '''
        Berechnet den Sinuswert (in Grad) des eingegebenen Winkels.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.sin(math.radians(worth))))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def cosinus(self):
        '''
        Berechnet den Kosinuswert (in Grad) des eingegebenen Winkels.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.cos(math.radians(worth))))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def tangens(self):
        '''
        Berechnet den Tangenswert (in Grad) des eingegebenen Winkels.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.tan(math.radians(worth))))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def log10(self):
        '''
        Berechnet den dekadischen Logarithmus (log10) des eingegebenen Wertes.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.log10(worth)))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def ln(self):
        '''
        Berechnet den natürlichen Logarithmus (ln) des eingegebenen Wertes.
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.log(worth)))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def one_through(self):
        '''
        Berechnet den Kehrwert des eingegebenen Wertes (1/x).
        '''
        try:
            worth = float(self.entry.get())
            self.entry.delete(0, END)
            self.entry.insert(0, str(1 / worth))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")

    def factorial(self):
        '''
        Berechnet die Fakultät des eingegebenen Wertes (nur für ganze Zahlen 
        geeignet).
        '''
        try:
            worth = int(float(self.entry.get()))
            self.entry.delete(0, END)
            self.entry.insert(0, str(math.factorial(worth)))
        except:
            self.entry.delete(0, END)
            self.entry.insert(0, "Fehler")