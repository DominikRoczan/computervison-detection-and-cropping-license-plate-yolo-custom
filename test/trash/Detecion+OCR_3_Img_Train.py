import os
from PIL import Image
from import_image import image_path

# Ścieżka do wytrenowanego modelu YOLOv5 do detekcji tablic rejestracyjnych
plate_model_path = 'D:/Machine_Learning/Projekty/01_ObjectDetection-CarLicensePlates/trained_model/train7/weights/best.pt'
images_path = image_path

if os.path.exists(plate_model_path and images_path):
    print("Plik modelu istnieje.")
else:
    print("Plik modelu nie istnieje.")

from ultralytics import YOLO
import os
import easyocr

# Ścieżka do pliku zawierającego wytrenowany model
model = YOLO(plate_model_path)  # load model
detection = model(images_path)  # detection

dir_name = '../../results'
os.makedirs(dir_name, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

# Otwórz plik tekstowy do zapisu wyników OCR
results_txt_path = os.path.join(dir_name, 'detection_results.txt')
results_txt_file = open(results_txt_path, 'w')

for i, det in enumerate(detection):
    # Wyodrębnij obrazy obszarów z tablicami rejestracyjnymi
    plate_images = [det[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in det.xyxy[0] if box[-1] == 0]

    # Zapisz każdy obszar z tablicą rejestracyjną do pliku
    for j, plate_img in enumerate(plate_images):
        plate_img_path = os.path.join(dir_name, f"detection_{i}_plate_{j}.jpg")
        Image.fromarray(plate_img).save(plate_img_path)
        print(f"Saved plate detection results to: {plate_img_path}")

        # Wykonaj OCR na obrazie z tablicą rejestracyjną
        ocr_result = reader.readtext(plate_img)

        # Zapisz wyniki OCR do pliku tekstowego
        results_txt_file.write(f"Detection {i} - Plate {j}:\n")
        for bbox, text, prob in ocr_result:
            results_txt_file.write(f"Text: {text}, Probability: {prob}\n")

# Zamknij plik tekstowy
results_txt_file.close()
