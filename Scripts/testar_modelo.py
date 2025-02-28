import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
load_model = keras.models.load_model

modelo = load_model("Models/modelo_libras.h5")

# Função para pré-processar uma imagem antes de passar para o modelo
def preprocessar_imagem(caminho_imagem, tamanho=(128, 128)):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError("Erro ao carregar a imagem. Verifique o caminho.")
    
    imagem = cv2.resize(imagem, tamanho)  # Redimensiona para o tamanho do modelo
    imagem = imagem / 255.0  # Normaliza os pixels (0 a 1)
    imagem = np.expand_dims(imagem, axis=0)  # Adiciona uma dimensão para compatibilidade
    return imagem

# Caminho de teste (coloque uma imagem real do dataset para testar)
caminho_teste = "DataSet/Turismo/ANEL-VIARIO/frame_0.png/"  # Ajuste esse caminho!

# Pré-processar a imagem
imagem_teste = preprocessar_imagem(caminho_teste)

# Fazer a predição
predicao = modelo.predict(imagem_teste)

# Mostrar a classe prevista
classe_predita = np.argmax(predicao)
print(f"Classe predita: {classe_predita}")