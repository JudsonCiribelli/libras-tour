import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
load_model = keras.models.load_model
import os

# Carregar o modelo treinado
modelo = load_model("models/modelo_libras.h5")

# Diretório do dataset
dataset_path = "DataSet/Turismo/"
categorias = sorted(os.listdir(dataset_path))  # Pegamos as pastas com os nomes dos pontos turísticos

# Escolher uma imagem de teste
caminho_teste = os.path.join(dataset_path, categorias[0], "frame_1.png")  # Altere para uma imagem real

# Função para processar a imagem antes da predição
def preprocessar_imagem(caminho_imagem, tamanho=(128, 128)):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Erro ao carregar a imagem {caminho_imagem}. Verifique o caminho.")
    
    imagem = cv2.resize(imagem, tamanho)  # Redimensiona para o tamanho do modelo
    imagem = imagem / 255.0  # Normaliza os pixels (0 a 1)
    imagem = np.expand_dims(imagem, axis=0)  # Adiciona dimensão para compatibilidade
    return imagem

# Testar uma imagem real do dataset
imagem_teste = preprocessar_imagem(caminho_teste)

# Fazer a predição
predicao = modelo.predict(imagem_teste)
classe_predita = np.argmax(predicao)

# Exibir a predição com o nome correto
print(f"Classe predita: {classe_predita} ({categorias[classe_predita]})")
