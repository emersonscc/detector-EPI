# EPI Detector — Detecção de Máscara Facial e EPIs

Projeto da disciplina de Processamento de Imagens e Visão Computacional (UPF).

Aplicação que verifica, a partir de imagens ou vídeo em tempo real, se uma pessoa está usando equipamentos de proteção (máscara facial, óculos de segurança, capacete), sinalizando o resultado na imagem processada.

## Como funciona
1. **Detecção facial** — localiza os rostos na imagem (Haar Cascade / MediaPipe).
2. **Pré-processamento** — recorte, redimensionamento e normalização da região do rosto.
3. **Classificação** — modelo leve (ex.: MobileNetV2) indica presença ou ausência do EPI.
4. **Saída** — caixa colorida e rótulo sobre cada rosto detectado.

## Estrutura
```
epi-detector/
├── data/            # datasets (não versionados)
│   ├── raw/
│   └── processed/
├── docs/            # relatório e materiais da disciplina
├── models/          # modelos treinados (.h5/.keras)
├── notebooks/       # experimentos e análises
├── results/         # imagens/vídeos de saída
├── scripts/         # treino e preparação de dados
├── src/
│   ├── detection/       # detecção facial
│   ├── classification/  # classificador de EPI
│   ├── utils/           # funções auxiliares de imagem
│   └── main.py          # ponto de entrada
└── tests/
```

## Como executar
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.main --image caminho/para/imagem.jpg
python -m src.main --webcam
```

## Tecnologias
Python, OpenCV, MediaPipe, TensorFlow/Keras, NumPy.
