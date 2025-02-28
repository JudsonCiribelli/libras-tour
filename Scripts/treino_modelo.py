import sys
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import carregar_imagens, dividir_dataset

# Definições de camadas da CNN
Sequential = keras.models.Sequential
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
to_categorical = tf.keras.utils.to_categorical
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Criar objeto de Data Augmentation com ajustes aprimorados
data_augmentation = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Diretório do dataset
diretorio_dataset = "DataSet/Turismo/"
imagens, labels = carregar_imagens(diretorio_dataset)

# Normalizar pixels (0 a 1)
imagens = imagens.astype('float32') / 255.0

# Transformar labels em categorias (one-hot encoding)
unique_labels = sorted(set(labels))
label_dict = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_dict[label] for label in labels]).astype(np.int32)
labels = to_categorical(labels, num_classes=len(unique_labels))

# Criar o dicionário de mapeamento de classes
mapeamento_classes = {idx: label for idx, label in enumerate(unique_labels)}

# Salvar o mapeamento em um arquivo JSON
with open("mapeamento_classes.json", "w") as f:
    json.dump(mapeamento_classes, f)

print("Mapeamento de classes salvo em 'mapeamento_classes.json'")

# Verificar o mapeamento das classes
with open("mapeamento_classes.json", "r") as f:
    mapeamento = json.load(f)
print("Mapeamento de Classes:", mapeamento)

# Dividir dataset
X_treino, X_val, X_teste, y_treino, y_val, y_teste = dividir_dataset(imagens, labels)

# Criar modelo CNN otimizado com Transfer Learning
def criar_modelo(input_shape, num_classes):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congelar os pesos do modelo base
    
    modelo = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Aprimorando taxa de aprendizado
    modelo.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return modelo

modelo = criar_modelo(X_treino.shape[1:], len(unique_labels))

# Callbacks para salvar o melhor modelo e interromper cedo
checkpoint = ModelCheckpoint(
    "Models/melhor_modelo_libras.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Treinar o modelo usando data augmentation
modelo.fit(
    data_augmentation.flow(X_treino, y_treino, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=60,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Avaliar no conjunto de teste
loss, accuracy = modelo.evaluate(X_teste, y_teste, verbose=1)
print(f"Perda no teste: {loss:.4f}")
print(f"Acurácia no teste: {accuracy:.4f}")

np.save("models/labels.npy", np.argmax(y_treino, axis=1))
print("Arquivo labels.npy salvo com sucesso!")

# Salvar modelo final
modelo.save("models/modelo_libras.h5")
print("Modelo salvo na pasta 'models'!")
