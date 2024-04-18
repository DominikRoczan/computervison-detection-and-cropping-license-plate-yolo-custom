'główny: responsywne labele+wycinanie tablic+biały pasek+ prwidłowo wklejone tablice'
from ultralytics import YOLO
import cv2
import numpy as np
import os

from import_image import image_path  # Zaimportuj ścieżkę do obrazu

# Lista obrazów do przetworzenia
image_paths = image_path
image = cv2.imread(image_paths)

# Załaduj model YOLO
model_path = 'D:/Machine_Learning/Projekty/01_ObjectDetection-CarLicensePlates/trained_model/train7/weights/best.pt'
model = YOLO(model_path)  # Używając ścieżki do wcześniej wytrenowanego modelu

# Przeprowadź detekcję na listę obrazów
results = model(image_paths)  # Zwraca listę obiektów Results

# Utwórz folder na wyniki, jeśli nie istnieje
dir_name = '../../results'
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

idx = 0
detected = False  # Flaga, czy wykryto jakieś tablice
extension_width = 400  # Szerokość dodatkowego białego paska
cropped_images = []  # Lista na wycięte obrazy tablic



# Przetwarzanie wyników
for r in results:
    boxes = r.boxes
    class_names = list(r.names.values())

    # Iteruj przez każde pole (bounding box)
    for b in boxes:

        xyxy = b.xyxy  # Zakładamy, że xyxy to tensor z 4 wartościami [x1, y1, x2, y2]
        confidence = float(b.conf)  # Prawidłowa pewność detekcji

        class_id = b.cls.item()  # Get class ID for each box
        class_name = r.names[class_id]  # Convert class ID to class name

        label = f"{class_name} {confidence:.2f}"

        if confidence > 0.5:
            detected = True
            box_width = int(xyxy[0][2]) - int(xyxy[0][0])
            idx += 1

            # Skaluj tekst, aby był 1.5x szerszy niż bounding box
            label_width = box_width * 1.5
            # Początkowy rozmiar czcionki
            font_scale = 0.5
            # Oblicz rozmiar tekstu z początkowym rozmiarem czcionki
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            # Skaluj font_scale do osiągnięcia żądanej szerokości etykiety
            font_scale *= label_width / text_size[0]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            # Ustawienie nowych współrzędnych tła tekstu
            text_x = int(xyxy[0][0]) - 1
            text_y = int(xyxy[0][1])
            background_tl = (text_x, text_y - text_size[1])
            background_br = (text_x + text_size[0], text_y)

            # Rysowanie etykiety
            cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), (0, 0, 255), 2)
            cv2.rectangle(image, background_tl, background_br, (0, 0, 255), cv2.FILLED)
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

            # cropped plates
            cropped_image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
            cropped_image_path = os.path.join(dir_name, f'cropped_plate_{idx}.jpg')
            cv2.imwrite(cropped_image_path, cropped_image)

            # Wycięcie i zapisanie tablicy rejestracyjnej
            x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

if detected:
    # Obliczenie wymiarów tablic rejestracyjnych
    target_width = int(0.75 * extension_width)
    resized_images = []
    for img in cropped_images:
        scale_factor = target_width / img.shape[1]
        new_height = int(img.shape[0] * scale_factor)
        resized_image = cv2.resize(img, (target_width, new_height))
        resized_images.append(resized_image)

    # Utworzenie rozszerzonego obrazu z dodatkowym białym paskiem
    height, width, channels = image.shape
    extended_image = np.full((height, width + extension_width, channels), 255, np.uint8)
    extended_image[:, :width] = image

    # Równomierne rozmieszczenie wyciętych tablic
    total_height_used = sum(img.shape[0] for img in resized_images)
    spacing = (height - total_height_used) // (len(resized_images) + 1)

    # Umieszczanie wyciętych tablic względem osi pionowej
    current_y = spacing
    for img in resized_images:
        start_x = width + (extension_width - target_width) // 2  # Obliczanie pozycji startowej na osi X
        extended_image[current_y:current_y + img.shape[0], start_x:start_x + img.shape[1]] = img
        current_y += img.shape[0] + spacing

    final_image_path = os.path.join(dir_name, 'detected_plat_extended.jpg')
    cv2.imwrite(final_image_path, extended_image)
    cv2.imshow('Detected Image with Plates', extended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    final_image_path = os.path.join(dir_name, 'detected_plat.jpg')
    cv2.imwrite(final_image_path, image)
    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

