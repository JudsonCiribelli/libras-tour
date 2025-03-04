import os
import cv2
import numpy as np

def carregar_imagens(diretorio, tamanho=(224, 224)):
    imagens = []
    labels = []
    for label in os.listdir(diretorio):
        pasta_classe = os.path.join(diretorio, label)
        if os.path.isdir(pasta_classe):
            for arquivo in os.listdir(pasta_classe):
                caminho_imagem = os.path.join(pasta_classe, arquivo)
                imagem = cv2.imread(caminho_imagem)
                if imagem is not None:
                    imagem = cv2.resize(imagem, tamanho)
                    imagens.append(imagem)
                    labels.append(label)
    return np.array(imagens), np.array(labels)