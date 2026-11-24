"""Aba de pré-processamento: predefinições, filtros ajustáveis e histograma de luminância."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from src.gui.widgets import section
from src.preprocess import PRESETS, PreprocessSettings, luminance_histogram

CUSTOM = "Personalizado"

# (campo liga/desliga, campo do valor, nome, dica, mínimo, máximo, escala do slider, formato)
FILTERS = [
    ("white_balance", None, "Balanço de branco", "Gray world: remove a dominante de cor da iluminação.",
     0, 0, 1, None),
    ("gamma_on", "gamma", "Correção de gama", "Abaixo de 1 clareia as sombras; acima de 1 escurece.",
     30, 250, 100, "{:.2f}"),
    ("clahe_on", "clahe_clip", "CLAHE (contraste local)", "Equaliza o histograma por blocos, no canal L (Lab).",
     10, 40, 10, "{:.1f}"),
    ("denoise_on", "denoise_strength", "Redução de ruído", "Filtro bilateral: suaviza e preserva as bordas.",
     10, 150, 1, "{:.0f}"),
    ("sharpen_on", "sharpen_amount", "Nitidez", "Máscara de nitidez (unsharp): realça as bordas.",
     2, 20, 10, "{:.1f}"),
]


class PreprocessPanel(QWidget):
    changed = Signal(object)   # PreprocessSettings

    def __init__(self):
        super().__init__()
        self._settings = PreprocessSettings()
        self._updating = False
        self.rows = {}

        # ---- predefinições
        self.preset = QComboBox(toolTip="Combinações prontas para situações comuns de captura")
        self.preset.addItems(list(PRESETS) + [CUSTOM])
        self.preset.currentTextChanged.connect(self._on_preset)
        btn_reset = QPushButton("Restaurar", toolTip="Desliga todos os filtros", clicked=lambda: self.preset.setCurrentText("Nenhum"))
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.preset, 1)
        tl.addWidget(btn_reset)

        # ---- um bloco por filtro: caixa de seleção + valor + slider + dica
        filters = QWidget()
        grid = QGridLayout(filters)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        row = 0
        for on_field, val_field, name, hint, lo, hi, scale, fmt in FILTERS:
            chk = QCheckBox(name)
            chk.toggled.connect(self._on_edit)
            value = QLabel("")
            value.setObjectName("confValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(chk, row, 0)
            grid.addWidget(value, row, 1)
            row += 1
            slider = None
            if val_field:
                slider = QSlider(Qt.Horizontal, minimum=lo, maximum=hi)
                slider.valueChanged.connect(self._on_edit)
                grid.addWidget(slider, row, 0, 1, 2)
                row += 1
            tip = QLabel(hint)
            tip.setObjectName("hint")
            tip.setWordWrap(True)
            grid.addWidget(tip, row, 0, 1, 2)
            row += 1
            grid.setRowMinimumHeight(row, 6)
            row += 1
            self.rows[on_field] = (chk, slider, value, val_field, scale, fmt)
        grid.setColumnStretch(0, 1)

        # ---- histograma
        self.hist = QLabel()
        self.hist.setAlignment(Qt.AlignCenter)
        self.hist.setMinimumHeight(100)
        legend = QLabel("<span style='color:#8a93a3'>■ original</span> &nbsp; "
                        "<span style='color:#e0a526'>■ processada</span> &nbsp; "
                        "<span style='color:#8a93a3'>(escuro → claro)</span>")
        legend.setObjectName("hint")
        self.prep_time = QLabel("Tempo do pré-processamento: —")
        self.prep_time.setObjectName("hint")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(section("Predefinição", top))
        lay.addWidget(section("Filtros", filters))
        lay.addWidget(section("Histograma de luminância", self.hist, legend, self.prep_time))
        lay.addStretch()

        self._apply_to_widgets(self._settings)

    # ---- estado <-> widgets
    def _apply_to_widgets(self, s):
        self._updating = True
        for on_field, (chk, slider, value, val_field, scale, fmt) in self.rows.items():
            chk.setChecked(getattr(s, on_field))
            if slider is not None:
                v = getattr(s, val_field)
                slider.setValue(int(round(v * scale)))
                slider.setEnabled(chk.isChecked())
                value.setText(fmt.format(v))
                value.setEnabled(chk.isChecked())
        self._updating = False

    def _read_widgets(self):
        changes = {}
        for on_field, (chk, slider, value, val_field, scale, fmt) in self.rows.items():
            changes[on_field] = chk.isChecked()
            if slider is not None:
                v = slider.value() / scale
                changes[val_field] = int(v) if val_field == "denoise_strength" else v
                slider.setEnabled(chk.isChecked())
                value.setText(fmt.format(v))
                value.setEnabled(chk.isChecked())
        return PreprocessSettings(**{**self._settings.__dict__, **changes})

    def _on_edit(self, *_):
        if self._updating:
            return
        self._settings = self._read_widgets()
        match = next((n for n, p in PRESETS.items() if p == self._settings), CUSTOM)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(match)
        self.preset.blockSignals(False)
        self.changed.emit(self._settings)

    def _on_preset(self, name):
        if name == CUSTOM:
            return
        self._settings = PRESETS[name]
        self._apply_to_widgets(self._settings)
        self.changed.emit(self._settings)

    @property
    def settings(self):
        return self._settings

    # ---- atualização a cada resultado
    def show_frame(self, result):
        w = max(self.hist.width() - 4, 200)
        img = luminance_histogram(result.original, result.processed, (w, 100))
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_BGR888).copy()
        self.hist.setPixmap(QPixmap.fromImage(qimg))
        self.prep_time.setText(f"Tempo do pré-processamento: {result.prep_ms:.0f} ms"
                               if self._settings.active else "Tempo do pré-processamento: — (sem filtros)")
