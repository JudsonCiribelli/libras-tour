import cv2
import mediapipe as mp
import os
import pandas as pd

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Diretório contendo as imagens organizadas nas 12 pastas
DATASET_PATH = "DATASET/Turismo/"  # <-- Substitua pelo caminho correto

# Lista para armazenar os dados extraídos
dataset = []

# Percorre todas as pastas (pontos turísticos)
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_path):
        
        # Percorre todas as imagens dentro da pasta
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Converte para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Processa a imagem com MediaPipe Hands
            result = hands.process(image_rgb)
            
            # Se houver detecção de mão
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for i, lm in enumerate(hand_landmarks.landmark):
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    
                    # Adiciona os pontos ao dataset
                    dataset.append([class_name] + landmarks)

# Ajusta as colunas corretamente
df = pd.DataFrame(dataset, columns=["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

# Salva os pontos extraídos em um CSV
df.to_csv("pontos_maos_bairros.csv", index=False)

print("Extração concluída! Dados salvos em pontos_maos.csv")
