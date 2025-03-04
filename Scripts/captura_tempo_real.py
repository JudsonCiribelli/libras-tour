import sys
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2

# Carregar modelo treinado
modelo_path = "Models/melhor_modelo_libras.h5"
if os.path.exists(modelo_path):
    modelo = load_model(modelo_path)
    print("Modelo carregado com sucesso!")
else:
    print("Modelo não encontrado. Certifique-se de treinar o modelo antes de executar este script.")
    sys.exit()

# Carregar o mapeamento de classes
mapeamento_path = "Models/mapeamento_classes.json"
if os.path.exists(mapeamento_path):
    with open(mapeamento_path, "r") as f:
        mapeamento_classes = json.load(f)
    print("Mapeamento de classes carregado.")
else:
    print("Erro: Mapeamento de classes não encontrado!")
    sys.exit()

# Configuração da webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    sys.exit()

print("Pressione 'q' para sair.")
ultimo_sinal = None
contador_frames = 0
frames_necessarios = 10  # Ajuste o número de frames para validação do sinal

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        continue
    
    # Pré-processamento da imagem
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_normalized = frame_resized.astype("float32") / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)
    
    # Fazer predição apenas após um número de frames consecutivos
    if contador_frames >= frames_necessarios:
        pred = modelo.predict(frame_input)
        classe_predita = np.argmax(pred)
        if classe_predita == ultimo_sinal:
            contador_frames += 1
        else:
            contador_frames = 0
        ultimo_sinal = classe_predita
        if contador_frames >= frames_necessarios:
            nome_classe = mapeamento_classes.get(str(classe_predita), "Desconhecido")
            contador_frames = 0  # Resetar contador após exibir predição
    
    # Exibir resultado na tela
    cv2.putText(frame, f"Predicao: {nome_classe}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Reconhecimento de Sinais", frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
