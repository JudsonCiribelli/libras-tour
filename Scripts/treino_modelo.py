import sys
import os
import json
# Adiciona o diretório raiz (PDI) ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import carregar_imagens, dividir_dataset
import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
import tensorflow as tf
MaxPooling2D = tf.keras.layers.MaxPooling2D
from tensorflow import keras
Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D 
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
import tensorflow as tf
to_categorical = tf.keras.utils.to_categorical
from utils.preprocess import carregar_imagens, dividir_dataset
import numpy as np
import tensorflow as tf
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping


# Adiciona o diretório raiz (PDI) ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import carregar_imagens, dividir_dataset

Sequential = keras.models.Sequential
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = keras.layers.Dense
Conv2D = keras.layers.Conv2D 
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Diretório do dataset
diretorio_dataset = "DataSet/Turismo/"
imagens, labels = carregar_imagens(diretorio_dataset)

# Normalizar pixels (0 a 1)
imagens = imagens / 255.0

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

# Dividir dataset
X_treino, X_val, X_teste, y_treino, y_val, y_teste = dividir_dataset(imagens, labels)

# Criar modelo CNN otimizado
def criar_modelo(input_shape, num_classes):
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

modelo = criar_modelo(X_treino.shape[1:], len(unique_labels))

# Callbacks para salvar o melhor modelo e interromper cedo
checkpoint = ModelCheckpoint(
    "Models/melhor_modelo_libras.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Treinar o modelo
modelo.fit(
    X_treino,
    y_treino,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Avaliar no conjunto de teste
loss, accuracy = modelo.evaluate(X_teste, y_teste, verbose=1)
print(f"Perda no teste: {loss:.4f}")
print(f"Acurácia no teste: {accuracy:.4f}")

np.save("models/labels.npy", np.argmax(y_treino, axis=1))  # Salva as labels convertidas de one-hot
print("Arquivo labels.npy salvo com sucesso!")

# Salvar modelo final
modelo.save("models/modelo_libras.h5")
print("Modelo salvo na pasta 'models'!")

