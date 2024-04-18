from ultralytics import YOLO
import cv2
import os
import pdb

from import_image import image_path  # Zaimportuj ścieżkę do obrazu

# Lista obrazów do przetworzenia
image_paths = image_path
image = cv2.imread(image_paths)

# Załaduj model YOLO
model_path = 'D:/Machine_Learning/Projekty/01_ObjectDetection-CarLicensePlates/trained_model/train7/weights/best.pt'
model = YOLO(model_path)  # Używając ścieżki do wcześniej wytrenowanego modelu

# Przeprowadź detekcję na listę obrazów
results = model(image_paths)  # Zwraca listę obiektów Results

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

        if confidence > 0.15:
            # Rysowanie ramki na obrazie

            cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), (0, 0, 255), 2)
            cv2.putText(image, label, (int(xyxy[0][0]), int(xyxy[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255),
                        1)

            # Utwórz folder na wyniki, jeśli nie istnieje
            dir_name = '../../results'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            final_image_path = os.path.join(dir_name, 'detected.jpg')
            cv2.imwrite(final_image_path, image)

            cv2.imshow('Detected Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
