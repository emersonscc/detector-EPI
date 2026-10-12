"""Ponto de entrada: processa imagem ou webcam."""
import argparse
import cv2

from src.detection.face_detector import FaceDetector
from src.classification.epi_classifier import EPIClassifier
from src.utils.image_utils import draw_result


def process_frame(frame, detector, classifier):
    for (x, y, w, h) in detector.detect(frame):
        label, conf = classifier.predict(frame[y:y + h, x:x + w])
        draw_result(frame, (x, y, w, h), label, conf, ok=(label == "com_epi"))
    return frame


def main():
    parser = argparse.ArgumentParser(description="Detector de EPI")
    parser.add_argument("--image", help="caminho da imagem")
    parser.add_argument("--webcam", action="store_true", help="usar webcam")
    args = parser.parse_args()

    detector, classifier = FaceDetector(), EPIClassifier()

    if args.image:
        frame = process_frame(cv2.imread(args.image), detector, classifier)
        cv2.imshow("EPI Detector", frame)
        cv2.waitKey(0)
    elif args.webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("EPI Detector", process_frame(frame, detector, classifier))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    else:
        parser.print_help()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
