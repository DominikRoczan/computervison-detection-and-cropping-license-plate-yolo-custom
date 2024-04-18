from ultralytics import YOLO  # Zaimportuj klasę YOLO z biblioteki ultralytics
from PIL import Image
import os
import easyocr  # Import biblioteki do OCR
from import_image import image_path  # Zaimportuj ścieżkę do obrazu

# Ścieżka do pliku zawierającego wytrenowany model
model_path = 'D:/Machine_Learning/Projekty/01_ObjectDetection-CarLicensePlates/trained_model/train7/weights/best.pt'

model = YOLO(model_path)  # Załaduj model

# Przeprowadź detekcję na obrazie

image = Image.open(image_path)  # Załaduj obraz
detection = model(image)  # Przeprowadź detekcję na obrazie

# Utwórz folder na wyniki, jeśli nie istnieje
dir_name = '../../results'
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

# Inicjalizacja OCR
reader = easyocr.Reader(['pl'], gpu=False)

# Przetwórz wykrycia
for i, det in enumerate(detection):  # Załóżmy, że detekcja jest iterowalna; dostosuj zgodnie z rzeczywistą strukturą
    img_path = os.path.join(dir_name, f'detected_{i}.jpg')
    # Zapisz obraz z wykryciem
    image.save(img_path)
    image.show()  # Wyświetl obraz

    # Wykonaj OCR na obrazie z wykryciem
    ocr_results = reader.readtext(img_path)
    # Zapisz wyniki OCR do pliku tekstowego
    with open(os.path.join(dir_name, f'ocr_results_{i}.txt'), 'w') as f:
        for bbox, text, prob in ocr_results:
            f.write(f"Detected text: {text}, Confidence: {prob}\n")
