"""Classificador de presença/ausência de EPI."""


class EPIClassifier:
    def __init__(self, model_path=None):
        self.model = None  # TODO: carregar modelo treinado (models/)

    def predict(self, face_img):
        """Retorna (rotulo, confianca). Placeholder até o modelo ser treinado."""
        # TODO: pré-processar e rodar o modelo
        return "indefinido", 0.0
