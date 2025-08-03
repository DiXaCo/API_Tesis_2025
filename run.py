import subprocess
import webbrowser
import threading

def abrir_navegador():
    webbrowser.open_new("http://localhost:5000")

threading.Timer(1.5, abrir_navegador).start()

subprocess.call(["python", "API/app.py"])
