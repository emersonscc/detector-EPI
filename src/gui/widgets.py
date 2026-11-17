"""Componentes visuais da interface."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

# cores da interface (as mesmas da anotação na imagem)
OK_COLOR = "#2e9e4f"
BAD_COLOR = "#d64545"
IDLE_COLOR = "#5b6472"

CLASS_INFO = [  # (classe, nome exibido, é violação?)
    ("helmet", "Com capacete", False),
    ("no_helmet", "Sem capacete", True),
    ("goggles", "Com óculos", False),
    ("no_goggles", "Sem óculos", True),
]


class ImageView(QLabel):
    """Mostra a imagem processada mantendo a proporção; aceita arrastar e soltar arquivos."""
    files_dropped = Signal(list)

    PLACEHOLDER = ("Arraste aqui uma imagem, um vídeo ou uma pasta\n\n"
                   "ou use os botões Imagem, Vídeo, Pasta ou Webcam acima")

    def __init__(self):
        super().__init__()
        self.setObjectName("imageView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self._pixmap = None
        self.clear_image()

    def clear_image(self):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self.PLACEHOLDER)

    def set_image(self, qimage):
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def _rescale(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragging", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragging", False)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("dragging", False)
        self.style().polish(self)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)


class StatusBadge(QLabel):
    """Faixa grande com a situação do quadro atual."""

    def __init__(self):
        super().__init__()
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(56)
        self.setWordWrap(True)
        self.set_state("aguardando")

    def set_state(self, state, violations=0, text=None):
        default = {
            "aguardando": ("Aguardando imagem", IDLE_COLOR),
            "carregando": ("Carregando modelo…", IDLE_COLOR),
            "ok": ("✔  EPI OK", OK_COLOR),
            "violacao": (f"⚠  ATENÇÃO\n{violations} item(ns) de EPI ausente(s)", BAD_COLOR),
            "vazio": ("Nenhuma pessoa detectada", IDLE_COLOR),
        }[state]
        self.setText(text or default[0])
        self.setStyleSheet(f"background:{default[1]};")


class ClassCounts(QWidget):
    """Contagem de detecções por classe, com a cor da anotação."""

    def __init__(self):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(6)
        self.values = {}
        for row, (cls, name, bad) in enumerate(CLASS_INFO):
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{BAD_COLOR if bad else OK_COLOR}; font-size:16px;")
            value = QLabel("0")
            value.setObjectName("countValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(dot, row, 0)
            grid.addWidget(QLabel(name), row, 1)
            grid.addWidget(value, row, 2)
            self.values[cls] = value
        grid.setColumnStretch(1, 1)

    def set_counts(self, counts):
        for cls, label in self.values.items():
            label.setText(str(counts.get(cls, 0)))


class InfoRow(QWidget):
    """Linha 'rótulo ........ valor' para o painel de desempenho."""

    def __init__(self, name):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel(name)
        title.setObjectName("infoName")
        self.value = QLabel("—")
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(title)
        lay.addStretch()
        lay.addWidget(self.value)

    def set(self, text):
        self.value.setText(text)


def section(title, *widgets):
    """Bloco do painel lateral com título."""
    frame = QFrame()
    frame.setObjectName("section")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(8)
    head = QLabel(title.upper())
    head.setObjectName("sectionTitle")
    lay.addWidget(head)
    for w in widgets:
        lay.addWidget(w)
    return frame
