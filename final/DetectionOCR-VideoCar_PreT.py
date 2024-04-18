'ładuje co 15 klatkę'
'jesli napis sie powtarza - nie zapisuj'

import torch
import cv2
import easyocr
import os
from import_image import video_path_2

# Załaduj wcześniej wytrenowany model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5n.pt')

def play_video(video_path_2, model):
    cap = cv2.VideoCapture(video_path_2)
    reader = easyocr.Reader(['en'], gpu=False)  # Inicjalizacja czytnika EasyOCR

    # Tworzenie folderu 'results', jeśli nie istnieje
    results_path = '../results'
    os.makedirs(results_path, exist_ok=True)

    # Tworzenie pliku tekstowego do zapisu wyników
    results_file = os.path.join(results_path, '01_detection_.txt')
    detected_texts = set()  # Zbiór przechowujący unikalne teksty
    frame_count = 0

    with open(results_file, 'w') as file:
        count_car = 0
        count_truck = 0
        count_bus = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 != 0:
                continue  # Przetwarzaj tylko co dziesiątą klatkę

            results = model(frame)

            for *xyxy, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]

                if label in desired_classes:
                    cropped_image = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    ocr_result = reader.readtext(cropped_image)
                    for (bbox, text, prob) in ocr_result:
                        if prob > 0.5 and text not in detected_texts:
                            detected_texts.add(text)
                            cv2.putText(frame, text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                            file.write(f'Detected text: {text} with confidence: {prob:.2f}\n')
                            print(f'Detected text: {text} with confidence: {prob:.2f}')

                    if label == 'car':
                        count_car += 1
                    elif label == 'truck':
                        count_truck += 1
                    elif label == 'bus':
                        count_bus += 1

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('Jedziemy!', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Zapisanie liczby wykrytych pojazdów na koniec pliku
        file.write(f'Total cars detected: {count_car}\n')
        file.write(f'Total trucks detected: {count_truck}\n')
        file.write(f'Total buses detected: {count_bus}\n')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desired_classes = ['car', 'truck', 'bus']
    play_video(video_path_2, model)
