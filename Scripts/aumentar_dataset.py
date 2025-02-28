import os
import cv2
import numpy as np
import imgaug.augmenters as iaa

# Configuração da aumentação
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% de chance de flip horizontal
    iaa.Affine(rotate=(-10, 10)),  # Rotação aleatória entre -10 e 10 graus
    iaa.GaussianBlur(sigma=(0, 1.0))  # Pequeno desfoque para variação
])

# Caminho da pasta original do dataset
dataset_path = "DataSet/Turismo/"
output_path = "DataSet/Turismo_Aumentado/"

# Criar pasta de saída se não existir
os.makedirs(output_path, exist_ok=True)

# Encontrar os nomes das classes automaticamente
todas_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Definir um limite para detectar classes desbalanceadas
limite_minimo = 100  # Ajuste conforme necessário
classes_desbalanceadas = []

for classe in todas_classes:
    input_dir = os.path.join(dataset_path, classe)
    num_imagens = len([f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")])

    if num_imagens < limite_minimo:  # Se a classe tem menos imagens que o mínimo, precisa de aumento
        classes_desbalanceadas.append(classe)

print(f"📌 Classes que precisam de aumento: {classes_desbalanceadas}")

# Agora aplicamos a aumentação apenas para essas classes
import imgaug.augmenters as iaa
import cv2

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% de chance de flip horizontal
    iaa.Affine(rotate=(-10, 10)),  # Rotação aleatória entre -10 e 10 graus
    iaa.GaussianBlur(sigma=(0, 1.0))  # Pequeno desfoque para variação
])

for classe in classes_desbalanceadas:
    input_dir = os.path.join(dataset_path, classe)
    output_dir = os.path.join(output_path, classe)
    os.makedirs(output_dir, exist_ok=True)

    imagens = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]

    if len(imagens) == 0:
        print(f"⚠️ Classe {classe} não tem imagens suficientes, pulando...")
        continue

    for img_name in imagens:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        for i in range(3):  # Criar 3 imagens aumentadas por imagem original
            augmented = seq(image=image)
            new_name = f"aug_{i}_{img_name}"
            cv2.imwrite(os.path.join(output_dir, new_name), augmented)

print("✅ Aumentação de imagens concluída!")
