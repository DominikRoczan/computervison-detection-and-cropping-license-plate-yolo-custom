'https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4'
import torch
from PIL import Image
import numpy as np
import easyocr
import os

# Załaduj wcześniej wytrenowany model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5n.pt')

# Wczytanie obrazu
image_path = 'D:/Machine_Learning/Projekty/01_ObjectDetection-CarLicensePlates/Recognition-CarLicensePlates/test/bmw.jpg'
image = Image.open(image_path)

# Przetwarzanie obrazu i detekcja
results = model(image)

# Wyniki detekcji
results_xyxy = results.xyxy[0]  # wyniki są w formacie [x1, y1, x2, y2, confidence, class]
reader = easyocr.Reader(['pl'], gpu=False)  # inicjalizacja EasyOCR

folder_path = 'result'
# Funkcja do zapewnienia istnienia folderu
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# Funkcja zapisywania wyników OCR do pliku
def save_ocr_results(ocr_results, output_folder, output_file):
    ensure_folder_exists(output_folder)
    with open(os.path.join(output_folder, output_file), 'w', encoding='utf-8') as file:
        for result in ocr_results:
            # location = result[0]
            detected_text = result[1]
            confidence = float(result[2])  # Przekonwertuj ufność na typ float
            # file.write(f"Detected text: \"{detected_text}\" ")
            file.write(f"Detected text: \"{detected_text}\" with confidence: {confidence:.4f}\n")


# Lista na wyniki OCR
ocr_results = []

# Iteracja przez wykryte obiekty i OCR
for result in results_xyxy:
    if result[-1] == 'car' or 'truck':  # Poprawka logiczna warunku
        x1, y1, x2, y2 = map(int, result[:4])
        cropped_image = image.crop((x1, y1, x2, y2))  # wycinanie obrazu do OCR
        ocr_result = reader.readtext(np.array(cropped_image), detail=1)  # OCR na wyciętym obrazie
        ocr_results.extend(ocr_result)
        print(ocr_result)  # Wyświetlenie wyników OCR

# Zapisz wyniki do pliku
output_folder = 'result'
output_file = 'img_ocr_results.txt'
save_ocr_results(ocr_results, output_folder, output_file)
