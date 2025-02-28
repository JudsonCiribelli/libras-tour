import sys
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from collections import deque

# Adiciona o diretório raiz (PDI) ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carregar modelo treinado
modelo = keras.models.load_model("models/modelo_libras.h5")

# Carregar mapeamento de classes
with open("mapeamento_classes.json", "r") as f:
    mapeamento_classes = json.load(f)
    mapeamento_classes = {int(k): v for k, v in mapeamento_classes.items()}  # Converter chaves para inteiros

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializar a webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erro ao acessar a webcam!")
    exit()

print("Pressione 'q' para sair.")

# Parâmetros para suavizar a predição
buffer_tamanho = 10  # Número de frames consecutivos para confirmar um sinal
predicoes_buffer = deque(maxlen=buffer_tamanho)
ultima_classe_exibida = None
frames_estaveis = 0
limiar_confianca = 0.75  # Apenas aceita previsões com confiança >= 75%

# Função para processar os frames antes da predição
def preprocessar_frame(frame, tamanho=(128, 128)):
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem = cv2.resize(imagem, tamanho)
    imagem = imagem / 255.0
    imagem = np.expand_dims(imagem, axis=0)
    return imagem

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame!")
        break

    # Definir um valor padrão para nome_categoria
    nome_categoria = "Aguardando sinal..."

    # Converter imagem para RGB para MediaPipe
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(imagem_rgb)

    # Verificar se alguma mão foi detectada
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Processar frame para predição
        imagem_processada = preprocessar_frame(frame)
        predicao = modelo.predict(imagem_processada)
        classe_predita = np.argmax(predicao)
        confianca = np.max(predicao)

        # Adiciona a classe detectada no buffer
        if confianca >= limiar_confianca:
            predicoes_buffer.append(classe_predita)
        
        # Verifica se o mesmo sinal foi detectado de forma consistente
        if len(predicoes_buffer) == buffer_tamanho and all(p == classe_predita for p in predicoes_buffer):
            if classe_predita != ultima_classe_exibida:
                ultima_classe_exibida = classe_predita
                nome_categoria = mapeamento_classes.get(classe_predita, "Desconhecido")
                frames_estaveis = 0  # Reinicia a contagem

        frames_estaveis += 1

    else:
        # Reinicia buffers quando nenhuma mão for detectada
        predicoes_buffer.clear()
        ultima_classe_exibida = None
        frames_estaveis = 0

    # Exibir resultado apenas quando um sinal for detectado consistentemente
    cv2.putText(frame, f"{nome_categoria}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Libras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
