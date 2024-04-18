
# Otwórz plik do zapisu wyników OCR
with open(ocr_output_path, 'w') as f:
    ocr_results = []  # Lista na wyniki OCR
    for cropped_image in cropped_images:
        # Odczytaj tekst z wyciętego obrazu
        result = ocr.readtext(cropped_image)
        ocr_results.append(result)  # Dodaj wynik do listy

        # Zapisz odczytany tekst do pliku
        f.write('Detected text:\n')
        for detection in result:
            f.write(f'{detection[1]}\n')
        f.write('\n')













# Dodanie tekstów pod tablicami na bocznym pasku
for cropped_image, endpoint, result in zip(cropped_images, label_endpoints, ocr_results):
    # Oblicz pozycję do umieszczenia tekstu pod tablicą
    text_position_x = int((k_1[0] + k_2[0]) / 2)  # Środek paska bocznego
    text_position_y = k_1[1] + 50  # 50 pikseli poniżej dolnej krawędzi białego paska


    # Konwertuj wynik OCR na ciąg tekstowy
    detected_text = '\n'.join(detection[1] for detection in result)

    # Dodaj tekst pod tablicą na bocznym pasku
    cv2.putText(extended_image, detected_text, (text_position_x, text_position_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),3)