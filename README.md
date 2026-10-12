# EPI Detector — Detecção de Capacete e Óculos de Proteção

Projeto da disciplina de Processamento de Imagens e Visão Computacional (UPF).

Aplicação que verifica, em imagens, vídeos ou webcam, se as pessoas estão usando **capacete** e **óculos de proteção**, sinalizando em verde o EPI presente e em vermelho o EPI ausente.

## Como funciona
Um detector de objetos **YOLO26n** (Ultralytics), ajustado por *transfer learning*, localiza e classifica quatro classes em cada frame:

| Classe | Significado |
|---|---|
| `helmet` | capacete presente |
| `no_helmet` | cabeça sem capacete |
| `goggles` | óculos de proteção presente |
| `no_goggles` | rosto sem óculos de proteção |

## Dataset
[PPEs — Roboflow Universe](https://universe.roboflow.com/ramcomills/ppes-kaxsi-v4bep) (~12 mil imagens, 12 classes, licença CC BY 4.0). O script `scripts/filter_dataset.py` mantém só as 4 classes acima e reindexa os rótulos.

## Estrutura
```
epi-detector/
├── data/                   # datasets (não versionados)
├── docs/                   # relatório
├── models/                 # pesos treinados (best.pt, não versionado)
├── notebooks/
│   └── treino_colab.ipynb  # treino completo no Google Colab
├── results/                # saídas anotadas
├── scripts/
│   └── filter_dataset.py   # filtragem e reindexação das classes
├── src/
│   ├── detector.py         # wrapper do modelo YOLO
│   ├── utils/drawing.py    # caixas, rótulos e faixa de status
│   └── main.py             # ponto de entrada
└── tests/
```

## Treino
Abra `notebooks/treino_colab.ipynb` no Google Colab com GPU e execute as células. Ao final, baixe o `best.pt` e coloque em `models/`.

## Execução local
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.main --image foto.jpg --save
python -m src.main --video obra.mp4
python -m src.main --webcam
```
Opções: `--conf 0.5` ajusta a confiança mínima; `--weights caminho.pt` usa outros pesos.

## Tecnologias
Python, Ultralytics YOLO26, OpenCV.
