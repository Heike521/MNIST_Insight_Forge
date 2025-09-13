# 🧮 Interaktiver Wissenschaftlicher Taschenrechner mit Farbwahl

Ein grafischer Taschenrechner in Python mit `customtkinter`, der neben den Grundrechenarten auch erweiterte mathematische Funktionen wie Sinus, Logarithmus, Fakultät und mehr unterstützt. Vor dem Start kann der Nutzer individuell die Farben für jede Funktionskategorie (z. B. Standard, wissenschaftlich, Analyse) anpassen.

---

## 📦 Funktionsumfang

- ✅ Addition, Subtraktion, Multiplikation, Division
- ✅ Prozentrechnung und Vorzeichenwechsel
- ✅ Klammerausdrücke
- ✅ Trigonometrische Funktionen: `sin`, `cos`, `tan` (in Grad)
- ✅ Logarithmen (`log`, `ln`) und Exponentialfunktionen
- ✅ Quadratwurzel, Potenz, Kehrwert `1/x`
- ✅ Fakultätsfunktion
- ✅ Farbauswahl je Funktionskategorie (Hintergrund/Text)
- ✅ Benutzerfreundliche GUI mit `CustomTkinter`

## 🚀 Startanleitung

### Voraussetzungen

- Python ≥ 3.9
- `customtkinter`  
  Installation mit:

  ```bash
  pip install customtkinter

📁 Taschenrechner-Projekt/
│
├── interaktiver_Taschenrechner.py    # Einstiegspunkt, startet GUI mit Farbwahl
├── Interface.py                      # Hauptklasse des Rechners (CustomTkinter-GUI)
├── CalculatorFunction.py             # Mathematische Logik (z. B. trigonometrische Funktionen)
├── ColorChoice.py                    # Vorschaltfenster zur Farbauswahl
├── ColorPicker.py                    # Hilfsfunktionen zum Farbwählen
├── README.md                         # Projektbeschreibung
├── .gitignore                        # Git-Konfigurationsdatei

## 👩‍💻 Autorin & Projektstand

**Autorin:** Heike Fasold  
**Projekt:** Interaktiver Taschenrechner mit Farbauswahl
**Stand:** 27. Juli 2025
