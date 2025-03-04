import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Adicionar o caminho do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar função de pré-processamento
from utils.preprocess import carregar_imagens

# Configurações
diretorio_dataset = "DataSet/Turismo/"
tamanho_imagem = (224, 224)
batch_size = 32
epochs = 50

# Carregar imagens e labels
imagens, labels = carregar_imagens(diretorio_dataset, tamanho_imagem)

# Verificar balanceamento das classes
print("Distribuição das classes:", Counter(labels))

# Normalizar pixels (0 a 1)
imagens = imagens.astype('float32') / 255.0

# Transformar labels em categorias (one-hot encoding)
unique_labels = sorted(set(labels))
label_dict = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_dict[label] for label in labels]).astype(np.int32)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(unique_labels))

# Salvar mapeamento de classes
os.makedirs("Models", exist_ok=True)
with open("Models/mapeamento_classes.json", "w") as f:
    json.dump({str(idx): label for idx, label in enumerate(unique_labels)}, f)

# Dividir dataset em treino (70%), validação (15%) e teste (15%)
X_treino, X_temp, y_treino, y_temp = train_test_split(imagens, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_teste, y_val, y_teste = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Criar modelo com EfficientNet
def criar_modelo(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    modelo = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return modelo

modelo = criar_modelo(X_treino.shape[1:], len(unique_labels))

# Callbacks
checkpoint = ModelCheckpoint("Models/melhor_modelo_libras.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# Balanceamento de classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_treino, axis=1)), y=np.argmax(y_treino, axis=1))
class_weights = dict(enumerate(class_weights))

# Treinamento
historico = modelo.fit(
    datagen.flow(X_treino, y_treino, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    class_weight=class_weights
)

# Avaliação
loss, accuracy = modelo.evaluate(X_teste, y_teste, verbose=1)
print(f"Acurácia no teste: {accuracy:.4f}")

# Matriz de Confusão
y_pred = modelo.predict(X_teste)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_teste, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# Salvar modelo
modelo.save("Models/modelo_libras.h5")
print("Modelo salvo na pasta 'Models'!")