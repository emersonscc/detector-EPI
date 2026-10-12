"""Funções auxiliares de processamento de imagem."""
import cv2


def preprocess_face(face_img, size=(224, 224)):
    face = cv2.resize(face_img, size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype("float32") / 255.0


def draw_result(image, box, label, confidence, ok=True):
    x, y, w, h = box
    color = (0, 200, 0) if ok else (0, 0, 220)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.0%}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image
