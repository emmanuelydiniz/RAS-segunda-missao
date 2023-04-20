import cv2
import mediapipe as mp

# a variavel cap pega da biblioteca cv2 a função videocapture, e retorna como um objeto para captura de imagem.
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# a variavel hans le a imagem rbg e retorna os marcos e a laterialidade de cada mão
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # se encontrar um marco na mão vai fazer tal procedimento
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(image, (cx, cy), 15, (173, 252, 3), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
