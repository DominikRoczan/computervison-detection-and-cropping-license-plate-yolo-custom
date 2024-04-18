import torch
import cv2

from import_image import video_path_2

# Wczytaj wytrenowany model YOLO

model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Przygotuj model do inferencji
model.eval()

# Przeniesienie modelu na dostępne urządzenie (GPU lub CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def play_video(video_path_2, model):
    cap = cv2.VideoCapture(video_path_2)
    count_car = 0
    count_truck = 0
    count_bus = 0

    while cap.isOpened():
        count_frame = 0
        ret, frame = cap.read()
        if not ret:
            break
        count_frame += 1
        # Konwersja klatki do formatu wymaganego przez YOLOv5 i przetwarzanie
        results = model(frame)


        # Iteracja przez detekcje i renderowanie wyników tylko dla wybranych klas
        for *xyxy, conf, cls in results.xyxy[0]:  # Tutaj iterujemy przez klasy detekcji
            label = model.names[int(cls)]  # Przekształcenie identyfikatora klasy na nazwę

            if label in desired_classes:  # Sprawdzenie, czy klasa detekcji jest wśród wybranych

                count_frame +=1
                if label == 'car':
                    count_car += 1
                elif label == 'truck':
                    count_truck += 1
                elif label == 'bus':
                    count_bus += 1
                # print(f'Frame: {count_frame}\nCar  : {count_car}\nTruck: {count_truck}\nBus  : {count_bus}\n')

                # Renderowanie detekcji na klatce
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                            2)

        cv2.imshow('Jedziemy!', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desired_classes = ['car', 'truck', 'bus']
    play_video(video_path_2, model)