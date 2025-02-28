# 📌 Reconhecimento de Sinais de Libras para Pontos Turísticos 🏛️✋

## 📌 Descrição do Projeto

Este projeto tem como objetivo **reconhecer sinais de Libras capturados por webcam e associá-los a pontos turísticos**. Para isso, utilizamos **redes neurais convolucionais (CNNs)** treinadas com imagens processadas pelo **MediaPipe** e aplicamos técnicas de aprendizado profundo para garantir uma detecção precisa.

A proposta foi desenvolvida dentro da disciplina **Processamento de Imagens**, ministrada pelo professor **Haroldo Gomes Barroso Filho**.

Com essa tecnologia, buscamos facilitar a comunicação e acessibilidade, permitindo que usuários possam **realizar sinais em Libras e obter informações sobre locais turísticos correspondentes em tempo real**.

---

## **📌 Metodologia**

O projeto segue um fluxo estruturado para a construção e avaliação do modelo de detecção de sinais:

### **1️⃣ Processamento das Imagens com MediaPipe**

- As imagens do dataset foram processadas utilizando **MediaPipe**, extraindo os **pontos de referência das mãos (landmarks)**.
- Cada imagem foi **convertida e normalizada**, garantindo um formato padronizado para o treinamento da rede neural.
- As imagens foram organizadas em **pastas separadas por classe**, onde cada classe representa um ponto turístico correspondente ao sinal realizado.

---

### **2️⃣ Construção do Modelo CNN**

- Utilizamos uma **Rede Neural Convolucional (CNN)** baseada em uma arquitetura otimizada para classificação de imagens.
- O modelo contém **camadas convolucionais e de pooling**, seguidas por **camadas densas e uma camada final com ativação softmax** para prever os sinais corretamente.
- O treinamento foi realizado utilizando a **função de perda Categorical Crossentropy** e o **otimizador Adam**, garantindo melhor convergência.

---

### **3️⃣ Treinamento e Avaliação**

- Utilizamos **Data Augmentation** para aumentar a diversidade dos dados e melhorar a robustez do modelo.
- O modelo foi treinado com **early stopping e checkpoints** para evitar overfitting.
- Durante a avaliação, foram analisadas **acurácia, matriz de confusão e métricas de precisão e recall**.

---

### **4️⃣ Captura e Inferência em Tempo Real**

- O sistema captura os sinais em Libras por meio da **webcam**, processa os frames em tempo real e realiza a predição.
- O modelo classifica o sinal detectado e **exibe o ponto turístico correspondente na tela**.
- Implementamos um **filtro de estabilidade**, garantindo que o modelo **espere um sinal completo** antes de exibir o resultado, evitando detecções errôneas.

---

## **📌 Como Executar o Projeto**

### **1️⃣ Instale as Dependências**

```bash
pip install -r requirements.txt
```
