import cv2
import mediapipe as mp
import numpy as np
import joblib
import subprocess
import tkinter as tk
from PIL import Image , ImageTk
from time import sleep

R = 0
#Carregar modelo treinado no arquivo treinamentos.py
modelo = joblib.load('modelo_gestos.pkl')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capturar vídeo da câmera
cap = cv2.VideoCapture(0)

def func1(frame):
    global R
    # Converter a imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processar a imagem com MediaPipe Hands
    resultados = hands.process(rgb_frame)

    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            # Desenhar landmarks na mão
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extrair coordenadas
            dados_mão = []
            for ponto in landmarks.landmark:
                dados_mão.extend([ponto.x, ponto.y, ponto.z])
            
            # Converter para numpy array e fazer a predição
            dados_mão = np.array(dados_mão).reshape(1, -1)
            gesto_previsto = modelo.predict(dados_mão)[0]
            # Exibir gesto reconhecido
            cv2.putText(frame, f'Gesto: {gesto_previsto}', (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            if gesto_previsto == 'PRIMEIRO' and R == 0:
                R = 1
                print('GESTO PRIMEIRO')
                subprocess.run(['git' , 'init'])
                subprocess.run(['git' , 'add' , '.'])
                subprocess.run(['git' , 'commit'  , '-m' , nome_do_commit.get()])
                subprocess.run(['git' , 'branch' , '-M' ,'main'])
                subprocess.run(['git' , 'remote' , 'add' ,'origin' , f'{nome_do_repo.get()}'])
                subprocess.run(['git' , 'push' , '-u' ,'origin'  , 'main'])
                print('Adicionando o comando' ,end='', flush=True)
                for i in range(10):
                    print('.' , end= '' , flush=True)
                    sleep(1)
            if gesto_previsto == 'POSITIVO':
                print()
                print('GESTO POSITIVO')
                subprocess.run(['git' , 'commit' , '-a' , '-m' , nome_do_commit.get()])
                subprocess.run(['git' , 'push'])
                print('Adicionando o comando' ,end='', flush=True)
                for i in range(10):
                    print('.' , end= '' , flush=True)
                    sleep(1)  
            if gesto_previsto == 'V':
                print()
                print('GESTO V')
                subprocess.run(['git' , 'checkout' , '-b' , nome_do_branch.get()])
                sleep(1)
                subprocess.run(['git' , 'commit' , '-a' , '-m' , nome_do_commit.get()])
                sleep(1)
                subprocess.run(['git' , 'push' , '--set-upstream' , 'origin' , nome_do_branch.get()])
                sleep(1)
                print('Adicionando o comando' ,end='', flush=True)
                for i in range(5):
                    print('.' , end= '' , flush=True)
                    sleep(1)  
            if gesto_previsto == 'VOLTAMAIN':
                print()
                print('GESTO VOLTAMAIN')
                subprocess.run(['git' , 'checkout' , 'main'])
                sleep(1)
                print('Adicionando o comando' ,end='', flush=True)
                for i in range(5):
                    print('.' , end= '' , flush=True)
                    sleep(1)
            if gesto_previsto == 'TROCAR':
                print()
                print('GESTO TROCAR')
                subprocess.run(['git' , 'checkout' , nome_do_branch.get()])
                sleep(1)
                print('Adicionando o comando' ,end='', flush=True)
                for i in range(5):
                    print('.' , end= '' , flush=True)
                    sleep(1)
    return frame

def func2():  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = func1(frame)
   
        # Exibir o frame
        cv2.imshow('Captura de Gestos', frame)
        tecla = cv2.waitKey(1) & 0xFF
        # Sair ao pressionar 'q'
        if tecla == ord('q'):
            cv2.destroyAllWindows()
            break
        elif tecla == ord('d'):
            cap.release()
            cv2.destroyAllWindows()
            break
        # Fechar arquivo e liberar recursos
    nome_do_commit.delete(0 , tk.END)
    nome_do_branch.delete(0 , tk.END)
    

janela = tk.Tk()
janela.title('Controle de Git por gesto')
janela.geometry('600x500')

imagem = Image.open('VISAOcomputacional.jpg')
imagem = imagem.resize((600 , 500))
imagem_tk = ImageTk.PhotoImage(imagem)

label_imagem = tk.Label(janela , image=imagem_tk)
label_imagem.place(x=0 , y=0 , relwidth=1 , relheight=1)


label_bemvindo = tk.Label(janela , text='GitHand' , font=('Arial' , 30) , background='gray')
label_bemvindo.pack()
label1 = tk.Label(janela , text='Digite o link do repo do gitHub:' , font=('Arial' , 20) , background='gray')
label1.pack(pady=15)
nome_do_repo = tk.Entry(janela , font=('Arial' , 20))
nome_do_repo.pack(pady= 5)
label2 = tk.Label(janela , text='Digite o título do Commit:' , font=('Arial' , 20) , background='gray')
label2.pack(pady=15)
nome_do_commit = tk.Entry(janela , font=('Arial' , 20))
nome_do_commit.pack(pady= 5)
label3 = tk.Label(janela , text='Digite o título do Branch:' , font=('Arial' , 20) , background='gray')
label3.pack(pady=15)
nome_do_branch = tk.Entry(janela , font=('Arial' , 20))
nome_do_branch.pack(pady= 5)
gesto_button = tk.Button(janela , text='Gesto' , font=('Arial' , 20) , background='blue' ,command= func2,  height=5 , width=30)
gesto_button.pack(pady=10)
janela.mainloop()