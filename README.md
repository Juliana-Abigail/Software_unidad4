
| **Campo magnético** | Líneas de campo de un imán de barra, solenoide, espira, hilo recto y dos hilos paralelos. Mide B con un clic y calcula la fuerza entre hilos. |
| **Fuerza de Lorentz** | Trayectoria 3D de una partícula cargada en campos E y B (círculo, hélice, selector de velocidades, deriva E×B, parábola). Con animación. |

## Cómo ejecutarlo en Visual Studio Code

1. Instala **Python 3.10 o superior** (https://www.python.org; en Windows marca *Add Python to PATH*).
2. En VS Code instala la extensión **Python** (Microsoft).
3. `Archivo > Abrir carpeta…` y elige esta carpeta (`simulador_unidad4`).
4. Abre la terminal (`Ctrl + ñ`) y escribe:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (en Linux/Mac:  source .venv/bin/activate)
   pip install -r requirements.txt
   python app.py
   ```
   También puedes abrir `app.py` y pulsar **F5** 
