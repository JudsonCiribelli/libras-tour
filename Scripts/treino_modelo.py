import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Carregar os dados extraídos
df = pd.read_csv("Points/pontos_maos.csv")

# Separar features (X) e labels (y)
X = df.drop(columns=["label"]).values  # Pegamos apenas os pontos das mãos
y = df["label"].values  # Pegamos as classes (nomes dos pontos turísticos)

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar e treinar o modelo SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Criar e treinar o modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Fazer previsões
y_pred_svm = svm_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Avaliar os modelos
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Acurácia do SVM: {accuracy_svm * 100:.2f}%")
print(f"Acurácia do KNN: {accuracy_knn * 100:.2f}%")

print("\nRelatório de Classificação - SVM:\n", classification_report(y_test, y_pred_svm))
print("\nRelatório de Classificação - KNN:\n", classification_report(y_test, y_pred_knn))

# Salvar o melhor modelo
if accuracy_svm > accuracy_knn:
    joblib.dump(svm_model, "modelo_girias.pkl")
    print("Modelo SVM salvo como 'modelo_girias.pkl'")
else:
    joblib.dump(knn_model, "modelo_girias.pkl")
    print("Modelo KNN salvo como 'modelo_girias.pkl'")
