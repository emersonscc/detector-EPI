"""Detecção de rostos na imagem."""
import cv2


class FaceDetector:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)

    def detect(self, image):
        """Retorna lista de (x, y, w, h) para cada rosto encontrado."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
