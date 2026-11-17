"""Janela principal do EPI Detector.

Execução:  python -m src.gui      (ou  python app.py)
"""
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, QSize, Qt, QTimer, QTranslator, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSplitter, QStyle, QToolBar, QVBoxLayout,
                               QWidget)

from src.gui.styles import apply_theme
from src.gui.widgets import BAD_COLOR, CLASS_INFO, OK_COLOR, ClassCounts, ImageView, InfoRow, StatusBadge, section
from src.gui.workers import BatchWorker, ModelLoader, StreamWorker, to_qimage
from src.pipeline import Pipeline
from src.utils.io import IMAGE_EXTS, VIDEO_EXTS, is_image, is_video, list_images, read_image, write_image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
RESULTS_DIR = ROOT / "results"
NAMES_PT = {cls: name for cls, name, _ in CLASS_INFO}
IMG_FILTER = "Imagens (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXTS)) + ")"
VID_FILTER = "Vídeos (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXTS)) + ")"


class MainWindow(QMainWindow):
    def __init__(self, weights=DEFAULT_WEIGHTS):
        super().__init__()
        self.setWindowTitle("EPI Detector — capacete e óculos de proteção")
        self.resize(1280, 800)
        self.setMinimumSize(980, 640)

        self.pipeline = None
        self.mode = None            # None | 'image' | 'folder' | 'stream' | 'batch'
        self.current_frame = None   # frame original (modos imagem/pasta), para reprocessar
        self.current_result = None
        self.current_name = ""
        self.folder_files, self.folder_index = [], 0
        self.stream = None
        self.batch = None

        self._build_actions()
        self._build_ui()
        self._set_sources_enabled(False)
        self._update_controls()

        # reprocessa a imagem atual pouco depois de mexer no slider (evita travar ao arrastar)
        self.reprocess_timer = QTimer(self, singleShot=True, interval=120)
        self.reprocess_timer.timeout.connect(self._reprocess_current)

        self._load_model(Path(weights))

    # ------------------------------------------------------------------ construção da interface
    def _build_actions(self):
        st = self.style()
        icon = lambda sp: st.standardIcon(sp)  # noqa: E731
        self.act_image = QAction(icon(QStyle.SP_FileIcon), "Imagem", self, shortcut=QKeySequence.Open,
                                 toolTip="Abrir uma imagem (Ctrl+O)", triggered=self.open_image)
        self.act_video = QAction(icon(QStyle.SP_MediaPlay), "Vídeo", self,
                                 toolTip="Abrir um arquivo de vídeo", triggered=self.open_video)
        self.act_folder = QAction(icon(QStyle.SP_DirOpenIcon), "Pasta", self,
                                  toolTip="Abrir uma pasta e navegar pelas imagens (← →)", triggered=self.open_folder)
        self.act_webcam = QAction(icon(QStyle.SP_ComputerIcon), "Webcam", self,
                                  toolTip="Iniciar a webcam selecionada", triggered=self.open_webcam)
        self.act_pause = QAction(icon(QStyle.SP_MediaPause), "Pausar", self, shortcut=Qt.Key_Space,
                                 toolTip="Pausar/continuar (Espaço)", checkable=True, toggled=self.toggle_pause)
        self.act_stop = QAction(icon(QStyle.SP_MediaStop), "Parar", self, triggered=self.stop_all)
        self.act_save = QAction(icon(QStyle.SP_DialogSaveButton), "Salvar", self, shortcut=QKeySequence.Save,
                                toolTip="Salvar o resultado atual com a faixa de status (Ctrl+S)",
                                triggered=self.save_current)
        self.act_record = QAction(icon(QStyle.SP_DialogYesButton), "Gravar", self, checkable=True,
                                  toolTip="Gravar o vídeo anotado em results/", toggled=self.toggle_record)
        self.act_batch = QAction(icon(QStyle.SP_FileDialogDetailedView), "Processar pasta", self,
                                 toolTip="Processar todas as imagens de uma pasta e gerar relatório CSV",
                                 triggered=self.run_batch)

    def _build_ui(self):
        tb = QToolBar("Ações", movable=False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addAction(self.act_image)
        tb.addAction(self.act_video)
        tb.addAction(self.act_folder)
        tb.addSeparator()
        self.cam_combo = QComboBox(toolTip="Câmera a usar")
        self.cam_combo.addItems([f"Câmera {i}" for i in range(4)])
        tb.addWidget(self.cam_combo)
        tb.addAction(self.act_webcam)
        tb.addSeparator()
        tb.addAction(self.act_pause)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_save)
        tb.addAction(self.act_record)
        tb.widgetForAction(self.act_record).setObjectName("recordButton")
        tb.addSeparator()
        tb.addAction(self.act_batch)

        # ---- área da imagem + barra de navegação
        self.view = ImageView()
        self.view.files_dropped.connect(self.open_paths)

        self.btn_prev = QPushButton("◀", toolTip="Imagem anterior (←)", clicked=lambda: self.step_folder(-1))
        self.btn_next = QPushButton("▶", toolTip="Próxima imagem (→)", clicked=lambda: self.step_folder(1))
        for b in (self.btn_prev, self.btn_next):
            b.setFixedWidth(44)
        self.lbl_name = QLabel("")
        self.lbl_name.setObjectName("fileName")
        self.progress = QProgressBar(textVisible=True)
        self.progress.setFixedWidth(240)
        self.progress.hide()
        nav = QHBoxLayout()
        nav.setContentsMargins(4, 0, 4, 0)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_name, 1)
        nav.addWidget(self.progress)
        nav.addWidget(self.btn_next)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 6, 12)
        lv.setSpacing(8)
        lv.addWidget(self.view, 1)
        lv.addLayout(nav)

        # ---- painel lateral
        self.badge = StatusBadge()
        self.counts = ClassCounts()
        self.det_list = QListWidget()
        self.det_list.setFixedHeight(104)

        self.conf_slider = QSlider(Qt.Horizontal, minimum=10, maximum=90, value=50, singleStep=5, pageStep=10)
        self.conf_value = QLabel("0,50")
        self.conf_value.setObjectName("confValue")
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_row = QWidget()
        cr = QHBoxLayout(conf_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.addWidget(QLabel("Confiança mínima"))
        cr.addStretch()
        cr.addWidget(self.conf_value)
        conf_hint = QLabel("Mais alto: menos alarmes falsos. Mais baixo: detecta mais, com mais erros.")
        conf_hint.setObjectName("hint")
        conf_hint.setWordWrap(True)
        self.chk_conf = QCheckBox("Mostrar a confiança nas caixas", checked=True)
        self.chk_conf.toggled.connect(self.on_show_conf)

        self.info_source = InfoRow("Fonte")
        self.info_size = InfoRow("Resolução")
        self.info_infer = InfoRow("Inferência")
        self.info_fps = InfoRow("Quadros/s")

        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(6, 12, 12, 12)
        pv.setSpacing(8)
        pv.addWidget(section("Situação", self.badge))
        pv.addWidget(section("Contagem", self.counts))
        pv.addWidget(section("Detecções", self.det_list))
        pv.addWidget(section("Ajustes", conf_row, self.conf_slider, conf_hint, self.chk_conf))
        pv.addWidget(section("Desempenho", self.info_source, self.info_size, self.info_infer, self.info_fps))
        pv.addStretch()
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(380)
        scroll.setFrameShape(QScrollArea.NoFrame)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.statusBar().showMessage("Carregando o modelo…")

    # ------------------------------------------------------------------ modelo
    def _load_model(self, weights):
        if not weights.is_file():
            path, _ = QFileDialog.getOpenFileName(self, "Selecione o modelo treinado (best.pt)",
                                                  str(ROOT), "Modelo YOLO (*.pt)")
            if not path:
                self.badge.set_state("aguardando", text="Modelo não encontrado")
                self.statusBar().showMessage(f"Modelo não encontrado em {weights}")
                return
            weights = Path(path)
        self.badge.set_state("carregando")
        self.loader = ModelLoader(str(weights), self.conf_slider.value() / 100)
        self.loader.loaded.connect(self._on_model_loaded)
        self.loader.failed.connect(lambda msg: QMessageBox.critical(self, "Erro ao carregar o modelo", msg))
        self.loader.start()
        self._weights_name = weights.name

    def _on_model_loaded(self, pipeline):
        self.pipeline = pipeline
        self.badge.set_state("aguardando")
        self._set_sources_enabled(True)
        self.statusBar().showMessage(f"Modelo {self._weights_name} carregado. Abra uma imagem, vídeo, pasta ou a webcam.")

    # ------------------------------------------------------------------ abrir fontes
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "", f"{IMG_FILTER};;Todos os arquivos (*)")
        if path:
            self.open_paths([path])

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir vídeo", "", f"{VID_FILTER};;Todos os arquivos (*)")
        if path:
            self.open_paths([path])

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Abrir pasta de imagens")
        if folder:
            self.open_paths([folder])

    def open_paths(self, paths):
        """Abre o que foi escolhido ou arrastado: pasta, vídeo, uma imagem ou várias imagens."""
        if self.pipeline is None:
            return
        paths = [Path(p) for p in paths]
        first = paths[0]
        if first.is_dir():
            self._start_folder(list_images(first), first.name)
        elif is_video(first):
            self._start_stream(first, is_webcam=False)
        elif len(paths) > 1 and all(is_image(p) for p in paths):
            self._start_folder(sorted(paths, key=lambda p: p.name.lower()), f"{len(paths)} imagens")
        elif is_image(first):
            self.stop_all()
            self.mode = "image"
            self._show_still(first)
        else:
            QMessageBox.warning(self, "Formato não suportado",
                                f"Não sei abrir '{first.name}'.\nImagens: {', '.join(sorted(IMAGE_EXTS))}\n"
                                f"Vídeos: {', '.join(sorted(VIDEO_EXTS))}")
        self._update_controls()

    def open_webcam(self):
        self._start_stream(self.cam_combo.currentIndex(), is_webcam=True)

    def _start_folder(self, files, label):
        self.stop_all()
        if not files:
            QMessageBox.information(self, "Pasta vazia", "Nenhuma imagem encontrada nessa pasta.")
            return
        self.mode = "folder"
        self.folder_files, self.folder_index = files, 0
        self.info_source.set(label)
        self._show_still(files[0])

    def step_folder(self, delta):
        if self.mode != "folder" or not self.folder_files:
            return
        self.folder_index = (self.folder_index + delta) % len(self.folder_files)
        self._show_still(self.folder_files[self.folder_index])

    def _show_still(self, path):
        try:
            frame = read_image(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Erro ao abrir", str(e))
            return
        self.current_frame, self.current_name = frame, Path(path).stem
        if self.mode == "image":
            self.info_source.set(Path(path).name)
        pos = f"   ({self.folder_index + 1}/{len(self.folder_files)})" if self.mode == "folder" else ""
        self.lbl_name.setText(Path(path).name + pos)
        self._reprocess_current()
        self._update_controls()

    def _reprocess_current(self):
        if self.current_frame is None or self.mode not in ("image", "folder"):
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = self.pipeline.process(self.current_frame)
        finally:
            QApplication.restoreOverrideCursor()
        self._show_result(to_qimage(result.annotated), result,
                          {"infer_ms": result.infer_ms, "size": result.original.shape[1::-1]})

    def _start_stream(self, source, is_webcam):
        self.stop_all()
        self.mode = "stream"
        self.current_name = f"webcam{source}" if is_webcam else Path(source).stem
        self.info_source.set(f"Câmera {source}" if is_webcam else Path(source).name)
        self.lbl_name.setText("Webcam ao vivo" if is_webcam else Path(source).name)
        self.stream = StreamWorker(self.pipeline, source, is_webcam)
        self.stream.frame_ready.connect(self._on_stream_frame)
        self.stream.message.connect(self.statusBar().showMessage)
        self.stream.finished_stream.connect(self._on_stream_finished)
        self.progress.setVisible(not is_webcam)
        self.progress.setValue(0)
        self.stream.start()
        self.statusBar().showMessage("Processando… (Espaço pausa)")
        self._update_controls()

    def _on_stream_frame(self, qimage, result, stats):
        self._show_result(qimage, result, stats)
        if stats.get("total"):
            self.progress.setMaximum(stats["total"])
            self.progress.setValue(stats["frame"])
            self.progress.setFormat(f"%v / %m quadros")

    def _on_stream_finished(self, reason):
        self.statusBar().showMessage(reason)
        if reason.startswith("Não foi possível"):
            QMessageBox.warning(self, "Erro", reason)
        if self.act_record.isChecked():
            self.act_record.blockSignals(True)
            self.act_record.setChecked(False)
            self.act_record.blockSignals(False)
            self.act_record.setText("Gravar")
        if self.stream is not None:
            self.current_result = self.stream.last_result
        self.stream = None
        self.act_pause.setChecked(False)
        self._update_controls()

    # ------------------------------------------------------------------ exibição do resultado
    def _show_result(self, qimage, result, stats):
        self.current_result = result
        self.view.set_image(qimage)
        self.badge.set_state(result.status, result.violations)
        self.counts.set_counts(result.counts)

        self.det_list.clear()
        # violações primeiro, depois por confiança
        for d in sorted(result.detections, key=lambda d: (not d.is_violation, -d.confidence)):
            mark = "⚠" if d.is_violation else "✔"
            item = QListWidgetItem(f"{mark}  {NAMES_PT.get(d.label, d.label)}  —  {d.confidence:.0%}")
            item.setForeground(QColor(BAD_COLOR if d.is_violation else OK_COLOR))
            item.setToolTip(f"Caixa (x1, y1, x2, y2): {d.box}")
            self.det_list.addItem(item)
        if not result.detections:
            self.det_list.addItem(QListWidgetItem("Nenhuma detecção acima da confiança mínima"))

        w, h = stats.get("size", (0, 0))
        self.info_size.set(f"{w} × {h}")
        self.info_infer.set(f"{stats.get('infer_ms', result.infer_ms):.0f} ms")
        self.info_fps.set(f"{stats['fps']:.1f}" if "fps" in stats else "—")

    # ------------------------------------------------------------------ controles
    def toggle_pause(self, paused):
        if self.stream:
            self.stream.set_paused(paused)
            self.statusBar().showMessage("Pausado (Espaço continua)" if paused else "Processando…")

    def stop_all(self):
        for worker in (self.stream, self.batch):
            if worker is not None:
                worker.stop()
                worker.wait(3000)
        self.stream = self.batch = None
        self.act_pause.setChecked(False)
        self.progress.hide()
        self._update_controls()

    def on_conf_changed(self, value):
        self.conf_value.setText(f"{value / 100:.2f}".replace(".", ","))
        if self.pipeline:
            self.pipeline.conf = value / 100
            self.reprocess_timer.start()

    def on_show_conf(self, checked):
        if self.pipeline:
            self.pipeline.show_conf = checked
            self.reprocess_timer.start()

    def save_current(self):
        result = self.stream.last_result if self.stream else self.current_result
        if result is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode == "stream":  # durante vídeo/webcam: captura rápida, sem diálogo
            path = RESULTS_DIR / f"{self.current_name}_captura_{stamp}.jpg"
        else:
            suggested = str(RESULTS_DIR / f"{self.current_name}_epi.jpg")
            path, _ = QFileDialog.getSaveFileName(self, "Salvar resultado", suggested,
                                                  "JPEG (*.jpg);;PNG (*.png)")
            if not path:
                return
        write_image(path, Pipeline.with_banner(result))
        self.statusBar().showMessage(f"Salvo em {path}")

    def toggle_record(self, on):
        if not self.stream:
            return
        self.act_record.setText("Gravando" if on else "Gravar")
        if on:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            RESULTS_DIR.mkdir(exist_ok=True)
            self.stream.start_recording(RESULTS_DIR / f"{self.current_name}_epi_{stamp}.mp4")
        else:
            self.stream.stop_recording()

    def run_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Pasta com as imagens a processar")
        if not folder:
            return
        files = list_images(folder)
        if not files:
            QMessageBox.information(self, "Pasta vazia", "Nenhuma imagem encontrada nessa pasta.")
            return
        self.stop_all()
        out = RESULTS_DIR / f"lote_{Path(folder).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mode = "batch"
        self.info_source.set(f"Lote: {Path(folder).name}")
        self.batch = BatchWorker(self.pipeline, folder, out)
        self.batch.progress.connect(self._on_batch_progress)
        self.batch.preview.connect(lambda img, r: self._show_result(img, r, {"size": r.original.shape[1::-1]}))
        self.batch.done.connect(self._on_batch_done)
        self.progress.setMaximum(len(files))
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m imagens")
        self.progress.show()
        self.batch.start()
        self._update_controls()

    def _on_batch_progress(self, i, total, name):
        self.progress.setValue(i)
        self.lbl_name.setText(name)
        self.statusBar().showMessage(f"Processando {i} de {total}: {name}")

    def _on_batch_done(self, csv_path, s):
        self.batch = None
        self.mode = None
        self.progress.hide()
        self._update_controls()
        out_dir = str(Path(csv_path).parent)
        self.statusBar().showMessage(f"Lote concluído: {out_dir}")
        msg = QMessageBox(self)
        msg.setWindowTitle("Processamento concluído")
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"<b>{s['imagens']} imagens processadas</b><br><br>"
                    f"<span style='color:{OK_COLOR}'>EPI OK: {s['ok']}</span><br>"
                    f"<span style='color:{BAD_COLOR}'>Com EPI ausente: {s['violacao']}</span><br>"
                    f"Sem pessoas detectadas: {s['vazio']}"
                    + (f"<br>Arquivos com erro de leitura: {s['erros']}" if s["erros"] else "")
                    + f"<br><br>Imagens anotadas e <i>relatorio_epi.csv</i> salvos em:<br>{out_dir}")
        open_btn = msg.addButton("Abrir pasta", QMessageBox.ActionRole)
        msg.addButton("Fechar", QMessageBox.RejectRole)
        msg.exec()
        if msg.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))

    def _set_sources_enabled(self, enabled):
        for a in (self.act_image, self.act_video, self.act_folder, self.act_webcam, self.act_batch):
            a.setEnabled(enabled)
        self.view.setAcceptDrops(enabled)

    def _update_controls(self):
        streaming = self.stream is not None
        batching = self.batch is not None
        self.act_pause.setEnabled(streaming)
        self.act_stop.setEnabled(streaming or batching)
        self.act_record.setEnabled(streaming)
        self.act_save.setEnabled(not batching and (streaming or self.current_result is not None))
        self.act_save.setText("Capturar" if streaming else "Salvar")
        folder = self.mode == "folder" and len(self.folder_files) > 1
        self.btn_prev.setVisible(folder)
        self.btn_next.setVisible(folder)

    # ------------------------------------------------------------------ teclado e fechamento
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.step_folder(-1)
        elif event.key() == Qt.Key_Right:
            self.step_folder(1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stop_all()
        super().closeEvent(event)


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("EPI Detector")
    # botões padrão do Qt (OK, Cancelar…) em português
    translator = QTranslator(app)
    if translator.load(QLocale("pt_BR"), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(translator)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
