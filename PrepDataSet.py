import cv2
import mediapipe as mp
import numpy as np
import csv

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Abrir arquivo CSV para salvar dados dos gestos
arquivo_csv = open('dados_gestos.csv', 'a', newline='')
escritor_csv = csv.writer(arquivo_csv)


print("Digite o nome do gesto (positivo/hang_loose/etc): ")
rotulo = input().strip().upper()

# Capturar vídeo da câmera
cap = cv2.VideoCapture(0)

def func(frame):
    # Converter a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processar a imagem com MediaPipe Hands
    resultados = hands.process(rgb_frame)

    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            # Desenhar landmarks na mão
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
    return frame , resultados.multi_hand_landmarks


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame , landmarks2 = func(frame)

    tecla = cv2.waitKey(1) & 0xFF   
    if tecla == ord('k') and landmarks2:
        for landmarks3 in landmarks2:
         # Extrair coordenadas normalizadas dos 21 pontos
         dados_mão = []
         for ponto in landmarks3.landmark:
             dados_mão.extend([ponto.x, ponto.y, ponto.z])

         # Salvar no CSV
         escritor_csv.writerow([rotulo] + dados_mão)

    # Exibir o frame
    cv2.imshow('Captura de Gestos', frame)

    # Sair ao pressionar 'q'
    if tecla == ord('q'):
      break


# Fechar arquivo e liberar recursos
arquivo_csv.close()
cap.release()
cv2.destroyAllWindows()
