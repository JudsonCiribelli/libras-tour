from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import joblib
import traceback

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# 🔹 Carregar modelos treinados para cada categoria
try:
    modelos = {
        "turismo": joblib.load("Models/modelo_turismo.pkl"),
        "bairros": joblib.load("Models/modelo_bairros.pkl"),
        "girias": joblib.load("Models/modelo_girias.pkl"),
    }
    print("✅ Modelos carregados com sucesso!")
except Exception as e:
    print("❌ Erro ao carregar modelos:", str(e))

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "image" not in data or "categoria" not in data:
            return jsonify({"error": "Imagem ou categoria não enviada"}), 400

        categoria = data["categoria"]
        
        # 🔹 Verificar se a categoria é válida
        if categoria not in modelos:
            return jsonify({"error": "Categoria inválida"}), 400

        print(f"📌 Categoria selecionada: {categoria}")

        # 📷 Exibir tamanho da string base64
        print(f"📷 Tamanho da string base64 recebida: {len(data['image'])}")

        # Remover cabeçalho 'data:image/jpeg;base64,...' se existir
        if "," in data["image"]:
            data["image"] = data["image"].split(",")[1]

        # Decodificar imagem Base64
        try:
            image_data = base64.b64decode(data["image"])
            image = Image.open(BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("❌ Erro ao decodificar a imagem base64:", str(e))
            return jsonify({"error": "Erro ao processar a imagem. Formato inválido."}), 400

        # Processar a imagem com MediaPipe
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks = np.array(landmarks).reshape(1, -1)

                # 🔹 Usar o modelo correspondente à categoria escolhida
                modelo = modelos[categoria]
                predicao = modelo.predict(landmarks)[0]

                print(f"✅ Predição realizada para {categoria}: {predicao}")
                return jsonify({"prediction": predicao})

        print("⚠ Nenhum sinal identificado")
        return jsonify({"prediction": "Nenhum sinal identificado"})

    except Exception as e:
        print("❌ Erro ao processar a predição:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
