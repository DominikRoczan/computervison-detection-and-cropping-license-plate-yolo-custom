from test.trash.Detecion_ImgPlate_Train_04_main import *
import os
import cv2
import numpy as np

resized_images = []

target_width = int(0.75 * extension_width)

for cropped_image in cropped_images:
    scale_factor = target_width / cropped_image.shape[1]
    new_height = int(cropped_image.shape[0] * scale_factor)
    resized_image = cv2.resize(cropped_image, (target_width, new_height))
    resized_images.append(resized_image)

if detected:
    # Utworzenie rozszerzonego obrazu z dodatkowym białym paskiem
    height, width, channels = image.shape
    extended_image = np.full((height, width + extension_width, channels), 255, np.uint8)
    extended_image[:, :width] = image

    # Obliczenie przestrzeni do rozmieszczenia wyciętych obrazów
    total_height_of_plates = sum(img.shape[0] for img in resized_images)
    space_left = height - total_height_of_plates
    space_between_plates = space_left // (len(resized_images) + 1)

    # Równomierne rozmieszczenie wyciętych tablic rejestracyjnych na białym pasku
    current_y_position = space_between_plates
    for index, (resized_img, endpoint) in enumerate(zip(resized_images, label_endpoints)):
        image_top_y = current_y_position

        # Wklejenie obrazu na środku białego paska
        center_of_extension = width + extension_width // 2
        image_left_x = center_of_extension - target_width // 2

        extended_image[image_top_y:image_top_y + resized_img.shape[0],
        image_left_x:image_left_x + target_width] = resized_img

        # Współrzędne końca linii powinny wskazywać na środek lewej krawędzi wyciętego obrazu
        line_end_x = image_left_x  # x-coordinate for the left edge of the pasted plate
        line_end_y = image_top_y + resized_img.shape[0] // 2  # y-coordinate for the vertical center of the pasted plate

        # Rysowanie linii łączącej
        cv2.line(extended_image, (endpoint[0], endpoint[1]), (line_end_x, line_end_y), (0, 0, 255), 2)

        # Aktualizacja pozycji dla następnej tablicy
        current_y_position += resized_img.shape[0] + space_between_plates

    # Zapisywanie i wyświetlanie obrazu końcowego
    final_image_path = os.path.join(dir_name, 'detected_plate_extended.jpg')
    cv2.imwrite(final_image_path, extended_image)
    cv2.imshow('Detected Image with Plates and Lines', extended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Obsługa przypadku, gdy nie wykryto tablic
    print("No license plates were detected.")
    final_image_path = os.path.join(dir_name, 'original_image.jpg')
    cv2.imwrite(final_image_path, image)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
