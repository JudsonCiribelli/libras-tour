import tensorflow as tf
import numpy as np
import cv2
import os
import json
from tensorflow import keras

# Carregar o modelo treinado
modelo = keras.models.load_model("models/modelo_libras.h5")

# Carregar o mapeamento de classes salvo no treinamento
mapeamento_path = "mapeamento_classes.json"
if os.path.exists(mapeamento_path):
    with open("mapeamento_classes.json", "r") as f:
        mapeamento_classes = json.load(f)
else:
    raise ValueError("Erro: Arquivo 'mapeamento_classes.json' não encontrado!")

# Diretório do dataset
dataset_path = "DataSet/Turismo/"
categorias = sorted(os.listdir(dataset_path))  # Pegamos as pastas com os nomes dos pontos turísticos

# Escolher uma imagem de teste (verifica se o caminho existe)
pasta_teste = categorias[0]  # Pegando a primeira categoria como teste
imagem_teste_path = os.path.join(dataset_path, pasta_teste, "frame_74.png")

if not os.path.exists(imagem_teste_path):
    raise ValueError(f"Erro: Imagem de teste não encontrada no caminho {imagem_teste_path}")

# Função para processar a imagem antes da predição
def preprocessar_imagem(caminho_imagem, tamanho=(128, 128)):  # Alterado para 128x128
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Erro ao carregar a imagem {caminho_imagem}. Verifique o caminho.")
    
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  # Converter para RGB
    imagem = cv2.resize(imagem, tamanho)  # Redimensiona para 128x128 (conforme o modelo)
    imagem = imagem / 255.0  # Normaliza os pixels (0 a 1)
    imagem = np.expand_dims(imagem, axis=0)  # Adiciona dimensão para compatibilidade
    return imagem

# Processar e testar a imagem real do dataset
imagem_teste = preprocessar_imagem(imagem_teste_path)

# Fazer a predição
predicao = modelo.predict(imagem_teste)
classe_predita_idx = np.argmax(predicao)
confianca = np.max(predicao)  # Pega a confiança da predição

# Recuperar o nome correto da classe predita
classe_predita_nome = mapeamento_classes.get(str(classe_predita_idx), "Desconhecido")

# Exibir a predição com o nome correto e a confiança
print(f"Classe Predita: {classe_predita_idx} ({classe_predita_nome})")
print(f"Confiança: {confianca:.2f}")
