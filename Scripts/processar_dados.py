import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import carregar_imagens


# Carregar modelo
modelo = keras.models.load_model("models/modelo_libras.h5")

# Carregar algumas imagens do dataset
diretorio_dataset = "DataSet/Turismo/"
imagens, labels = carregar_imagens(diretorio_dataset)

# Selecionar uma imagem de teste
imagem_teste = imagens[0]
imagem_teste = np.expand_dims(imagem_teste, axis=0)  # Adiciona dimensão de batch

# Fazer predição
predicao = modelo.predict(imagem_teste)
classe_predita = np.argmax(predicao)
confianca = np.max(predicao)

print(f"Classe predita: {classe_predita} - Confiança: {confianca:.2f}")
