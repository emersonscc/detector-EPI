"""Tema escuro da interface (paleta Fusion + folha de estilo Qt)."""
from PySide6.QtGui import QColor, QPalette

BG = "#14171c"
PANEL = "#1c2027"
BORDER = "#2a303a"
TEXT = "#e4e7ec"
MUTED = "#8a93a3"
ACCENT = "#e0a526"   # amarelo "segurança do trabalho"

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px 8px; spacing: 4px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 8px; }}
QToolButton {{ background: transparent; color: {TEXT}; border: 1px solid transparent; border-radius: 6px;
               padding: 6px 10px; }}
QToolButton:hover {{ background: #262c35; border-color: {BORDER}; }}
QToolButton:checked {{ background: #3a2f12; border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton:disabled {{ color: #4d5562; }}
#recordButton:checked {{ background: #3d1616; border-color: #d64545; color: #ff6b6b; font-weight: 700; }}
QComboBox {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; min-width: 92px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: #2c3440; }}

#imageView {{ background: #0d0f12; border: 1px dashed {BORDER}; border-radius: 10px; color: {MUTED};
              font-size: 15px; }}
#imageView[dragging="true"] {{ border: 2px dashed {ACCENT}; color: {ACCENT}; }}
#fileName {{ color: {MUTED}; padding: 0 6px; }}

#section {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
#section QLabel {{ background: transparent; }}
#sectionTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
#statusBadge {{ color: white; font-size: 17px; font-weight: 700; border-radius: 8px; padding: 10px; }}
#countValue {{ font-size: 16px; font-weight: 700; }}
#confValue {{ color: {ACCENT}; font-weight: 700; font-size: 14px; }}
#infoName, #hint {{ color: {MUTED}; }}
#hint {{ font-size: 11px; }}

QListWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; }}
QListWidget::item {{ padding: 4px 2px; }}
QListWidget::item:selected {{ background: #262c35; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QCheckBox {{ background: transparent; spacing: 8px; }}

QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 10px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QProgressBar {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; text-align: center;
                height: 20px; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QScrollArea {{ background: {BG}; }}
QSplitter::handle {{ background: {BG}; width: 4px; }}
QMessageBox {{ background: {PANEL}; }}
QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; border: 1px solid {BORDER}; padding: 7px 14px;
                margin-right: 4px; border-radius: 6px; }}
QTabBar::tab:selected {{ color: {TEXT}; border-color: {ACCENT}; background: #262c35; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
#compareButton:checked {{ background: #3a2f12; border-color: {ACCENT}; color: {ACCENT}; }}
QCheckBox:disabled, QLabel:disabled, #confValue:disabled {{ color: #4d5562; }}
QSlider::sub-page:horizontal:disabled {{ background: #3a414c; }}
QSlider::handle:horizontal:disabled {{ background: #4d5562; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {MUTED}; border-radius: 3px; background: {BG}; }}
QCheckBox::indicator:hover {{ border-color: {TEXT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
"""


def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in [(QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, BG),
                        (QPalette.AlternateBase, PANEL), (QPalette.Text, TEXT), (QPalette.Button, PANEL),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT),
                        (QPalette.HighlightedText, "#000000"), (QPalette.ToolTipBase, PANEL),
                        (QPalette.ToolTipText, TEXT)]:
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
