'https://github.com/ultralytics/ultralytics/issues/7175'

import torch
from torchvision import transforms
import cv2
import easyocr
import os
import pdb

from import_image import video_path_2


# Importuj moduł YOLO z repozytorium YOLOv5
from yolov5.models.yolo import Model

# Załaduj wytrenowany model YOLOv5 do detekcji tablic rejestracyjnych
plate_model_path = ('G:/Mój dysk/10_Machine Learning/00_Projekty/Data_Sets/PlateCar-Data_set/train7/weights/best.pt')
plate_model = Model(cfg='yolov5/models/yolov5n.yaml')

# Wczytanie wytrenowanych wag modelu
state_dict = torch.load(plate_model_path)  # Wczytanie pliku .pt
plate_model.load_state_dict(state_dict, strict=False)
plate_model.eval()

# Załaduj pre-trenowany model YOLOv5 do detekcji pojazdów
vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', device='cpu')
vehicle_model.eval()

# Transformacja obrazu do tensora PyTorch
transform = transforms.Compose([
    transforms.ToTensor()
])


def pad(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image


def play_video(video_path_2):
    cap = cv2.VideoCapture(video_path_2)

    reader = easyocr.Reader(['en'], gpu=True)
    detected_texts = set()

    results_path = 'results'
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, 'detection_results.txt')

    with open(results_file, 'w') as file:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 45 != 0:
                continue

            frame_padded = pad(frame)
            frame_tensor = transform(frame_padded).unsqueeze(0).to('cpu')  # Dodaje wymiar batch i przekazuje do CPU
            # print(2, frame_tensor)
            vehicle_results = vehicle_model(frame_tensor)
            print(5, vehicle_results)
            # print(5, vehicle_model.names)




            for *xyxy, conf, cls in vehicle_results[0]:

                pdb.set_trace()  # debuger
                label = vehicle_model.names[int(cls)]
                # label = vehicle_model.names

                if label in ['car', 'truck', 'bus']:

                    cropped_image = frame_padded[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0]):int(xyxy[2])]
                    license_results = plate_model(cropped_image)

                    for *lp_xyxy, lp_conf, lp_cls in license_results.xyxy[0]:
                        lp_label = plate_model.names[int(lp_cls)]
                        if lp_label == 'license_plate':
                            license_cropped_image = cropped_image[int(lp_xyxy[1]):int(lp_xyxy[3]),
                                                    int(lp_xyxy[0]):int(lp_xyxy[2])]
                            ocr_result = reader.readtext(license_cropped_image)
                            for (bbox, text, prob) in ocr_result:
                                if prob > 0.5 and text not in detected_texts:
                                    detected_texts.add(text)
                                    file.write(f'Detected license plate text: {text} with confidence: {prob:.2f}/n')
                                    print(f'Detected license plate text: {text} with confidence: {prob:.2f}')

            cv2.imshow('Detection Results', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    play_video(video_path_2)
