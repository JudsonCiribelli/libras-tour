import tkinter as tk
from tkinter import messagebox
import subprocess

# Função para iniciar a detecção de turismo
def iniciar_turismo():
    try:
        subprocess.Popen(["python", "Scripts/captura_tempo_real.py"])
    except Exception as e:
        messagebox.showerror("Erro", f"Não foi possível iniciar a detecção: {str(e)}")

# Criar a janela principal
root = tk.Tk()
root.title("Sistema de Detecção de Libras")
root.geometry("500x400")
root.configure(bg="#1e1e1e")

# Estilos
button_style = {
    "font": ("Poppins", 14, "bold"),
    "width": 20,
    "height": 2,
    "fg": "white",
    "bd": 0,
    "relief": "flat"
}

title_label = tk.Label(root, text="Sistema de Reconhecimento de Sinais", font=("Poppins", 16, "bold"), fg="white", bg="#1e1e1e")
title_label.pack(pady=20)

# Criar os botões
btn_turismo = tk.Button(root, text="Turismo", command=iniciar_turismo, bg="#3498db", **button_style)
btn_turismo.pack(pady=10)

btn_girias = tk.Button(root, text="Gírias", state=tk.DISABLED, bg="#7f8c8d", **button_style)
btn_girias.pack(pady=10)

btn_etc = tk.Button(root, text="Bairros", state=tk.DISABLED, bg="#7f8c8d", **button_style)
btn_etc.pack(pady=10)

# Rodapé
footer_label = tk.Label(root, text="Desenvolvido por Judson Ciribelli, Paulo Gabriel, Daniel Aires", font=("Arial", 10), fg="#95a5a6", bg="#1e1e1e")
footer_label.pack(side="bottom", pady=10)

# Iniciar a interface
tk.mainloop()
