import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def carregar_imagens(diretorio, tamanho_imagem=(128, 128)):
    imagens = []
    labels = []
    
    for categoria in os.listdir(diretorio):
        caminho_categoria = os.path.join(diretorio, categoria)
        
        if not os.path.isdir(caminho_categoria):
            continue

        for arquivo in os.listdir(caminho_categoria):
            caminho_arquivo = os.path.join(caminho_categoria, arquivo)
            
            imagem = cv2.imread(caminho_arquivo)
            if imagem is not None:
                imagem = cv2.resize(imagem, tamanho_imagem)
                imagens.append(imagem)
                labels.append(categoria)
    
    return np.array(imagens), np.array(labels)

def dividir_dataset(imagens, labels, test_size=0.2, val_size=0.1):
    X_treino, X_teste, y_treino, y_teste = train_test_split(imagens, labels, test_size=test_size, random_state=42)
    X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, test_size=val_size, random_state=42)
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste

# Criar um gerador de Data Augmentation
