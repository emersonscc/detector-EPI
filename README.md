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

### Dataset próprio (sintético + teste real)
O dataset do Roboflow não tem imagens parecidas com o uso real (pessoa de frente para a webcam). Para cobrir esse cenário:

1. **Recortes dos EPIs** — `scripts/prepare_cutouts.py` remove o fundo (liso ou xadrez de "transparência falsa") de fotos de capacetes e óculos e gera PNGs com transparência em `assets/epi/`. Usa segmentação por cor (HSV/Lab), análise de textura (desvio padrão local), componentes conexos e morfologia; lentes transparentes ficam semitransparentes.
2. **Captura** — `scripts/capture_webcam.py --mode sintese` grava frames do rosto **sem EPI**, descartando frames repetidos; `--mode teste` tira fotos **com EPI real** para o conjunto de teste (anotado à mão).
3. **Síntese** — `scripts/synth_dataset.py` detecta rosto e olhos (Haar), encaixa capacete e óculos por transformação afim, harmoniza cor/brilho, faz alpha blending e gera os rótulos YOLO automaticamente, nas 4 combinações de EPI.

```bash
python scripts/prepare_cutouts.py --src fotos_epi --dst assets/epi
python scripts/capture_webcam.py --mode sintese --pessoa emerson
python scripts/synth_dataset.py --frames data/raw/webcam/sintese --out data/synthetic --preview 20
```

## Estrutura
```
epi-detector/
├── assets/epi/             # recortes PNG de capacetes e óculos (não versionados)
├── data/                   # datasets (não versionados)
├── docs/                   # relatório
├── models/                 # pesos treinados (best.pt)
├── notebooks/
│   └── treino_colab.ipynb  # treino completo no Google Colab
├── results/                # saídas anotadas
├── scripts/
│   ├── filter_dataset.py   # filtragem, reindexação e divisão do dataset
│   ├── prepare_cutouts.py  # recorte dos EPIs (fundo -> transparência)
│   ├── capture_webcam.py   # captura de fotos pela webcam
│   └── synth_dataset.py    # geração do dataset sintético
├── app.py                  # abre a interface gráfica
├── src/
│   ├── detector.py         # wrapper do modelo YOLO
│   ├── pipeline.py         # pré-processamento -> detecção -> desenho
│   ├── gui/                # interface gráfica (PySide6)
│   │   ├── main_window.py  # janela principal
│   │   ├── workers.py      # vídeo/webcam e lote em threads
│   │   ├── widgets.py      # componentes visuais
│   │   └── styles.py       # tema
│   ├── utils/drawing.py    # caixas, rótulos e faixa de status
│   ├── utils/io.py         # leitura/gravação (caminhos com acento no Windows)
│   └── main.py             # versão em linha de comando
└── tests/
```

## Treino
Abra `notebooks/treino_colab.ipynb` no Google Colab com GPU e execute as células. Ao final, baixe o `best.pt` e coloque em `models/`.

## Execução local
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Interface gráfica
```bash
python app.py
```
- **Fontes:** imagem, vídeo, pasta (navegação com ← →) ou webcam, pelos botões ou **arrastando o arquivo para a janela**. Formatos: JPG, PNG, BMP, TIFF, WEBP; MP4, AVI, MOV, MKV, WMV.
- **Painel:** situação (EPI OK / atenção / nenhuma pessoa), contagem por classe, lista de detecções, confiança mínima ajustável e desempenho (tempo de inferência e quadros/s).
- **Saídas** (em `results/`): salvar a imagem anotada, capturar quadro e gravar o vídeo anotado durante vídeo/webcam, e **Processar pasta**, que anota todas as imagens de uma pasta e gera `relatorio_epi.csv` com o status de cada uma.

### Linha de comando
```bash
python -m src.main --image foto.jpg --save
python -m src.main --video obra.mp4
python -m src.main --webcam
```
Opções: `--conf` ajusta a confiança mínima (padrão 0,5); `--weights caminho.pt` usa outros pesos.

## Tecnologias
Python, Ultralytics YOLO26, OpenCV, PySide6 (Qt).
