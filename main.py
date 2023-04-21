import cv2
import mediapipe as mp

# a variavel cap pega da biblioteca cv2 a função videocapture, e retorna como um objeto para captura de imagem.
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# a variavel hands lê a imagem rbg e retorna os marcos e a laterialidade de cada mão
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# utilizando a função read para a variável cap
while True:
    success, image = cap.read()
    # converte a imagem de BGR para RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # se encontrar um marco na mão :
    if results.multi_hand_landmarks:
        knuckles_position = []
        fingertips_position = []
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape  # formato da imagem
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                fingertips = [4, 8, 12, 16, 20]  # id das pontas
                knuckles = [2, 6, 10, 14, 18]  # id das juntas

                if id in fingertips:  # se o marco estiver dentro da lista vai criar um círculo.
                    fingertips_position.append((lm.x, lm.y))
                    cv2.circle(image, (cx, cy), 8, (173, 252, 3), cv2.FILLED)
                if id in knuckles:
                    knuckles_position.append((lm.x, lm.y))

            up_count = 0
            for i in range(1, 5):
                fingertip_y = fingertips_position[i][1]
                knuckle_y = knuckles_position[i][1]
                # verifica para cada dedo se a ponta tá acima da junta
                if fingertip_y < knuckle_y:
                    up_count += 1

            # verifica se o dedão está dobrado, usando a coordenada x
            if knuckles_position[0][0] < fingertips_position[0][0]:
                up_count += 1

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            cv2.putText(image, str(up_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (209, 80, 0, 255), 3)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
