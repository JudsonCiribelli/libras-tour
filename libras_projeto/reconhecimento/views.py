from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import os
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import joblib
import traceback
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import sklearn

# 🔹 Carregar modelos treinados
try:
  modelos = {
  "turismo": joblib.load(os.path.join(settings.BASE_DIR, "Models", "modelo_turismo.pkl")),
  "bairros": joblib.load(os.path.join(settings.BASE_DIR, "Models", "modelo_bairros.pkl")),
  "girias": joblib.load(os.path.join(settings.BASE_DIR, "Models", "modelo_girias.pkl")),
  }
  print("✅ Modelos carregados com sucesso!")
except Exception as e:
  print(f"❌ Erro ao carregar modelos: {e}")

# Inicializar MediaPipe Hands
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 📌 Rota principal
def home(request):
  return JsonResponse({"message": "API de Reconhecimento de Libras está rodando!"})

# 📌 API para processar a imagem
@csrf_exempt
@api_view(['POST'])
def predict(request):
  try:
    if request.method != "POST":
     return JsonResponse({"error": "Método não permitido"}, status=405)

    data = request.data

    if "image" not in data or "categoria" not in data:
     return JsonResponse({"error": "Imagem ou categoria não enviada"}, status=400)

    categoria = data["categoria"]

    if categoria not in modelos:
     return JsonResponse({"error": "Categoria inválida"}, status=400)

    print(f"📌 Categoria selecionada: {categoria}")

    image_base64 = data["image"].split(",")[-1]

    # Decodificar imagem Base64
    try:
     image_data = base64.b64decode(image_base64)
     image = Image.open(BytesIO(image_data)).convert("RGB")
     frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
     print("❌ Erro ao decodificar a imagem base64:", str(e))
     return JsonResponse({"error": "Erro ao processar a imagem."}, status=400)

    # Processar imagem com MediaPipe
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
     for hand_landmarks in result.multi_hand_landmarks:
      landmarks = []
      for lm in hand_landmarks.landmark:
        landmarks.append(lm.x)
        landmarks.append(lm.y)
      landmarks = np.array(landmarks).reshape(1, -1)

      modelo = modelos[categoria]
      predicao = modelo.predict(landmarks)[0]

      print(f"✅ Predição realizada para {categoria}: {predicao}")
      return JsonResponse({"prediction": predicao})
    print("⚠ Nenhum sinal identificado")
    return JsonResponse({"prediction": "Nenhum sinal identificado"})

  except Exception as e:
   print("❌ Erro ao processar a predição:")
   traceback.print_exc()
   return JsonResponse({"error": str(e)}, status=500)