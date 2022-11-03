from keras.models import load_model
import cv2
import numpy as np
import os
import pyautogui as p

class_map = {
    0 : 'Forward',
    1 : 'None',
    2 : 'Play/Pause',
    3 : 'Volume-Down',
    4 : 'Volume-Up'
}

def control(user_move):
    if user_move == 'Forward':
        p.press("right")

    elif user_move == 'Play/Pause':
        p.press('space')

    elif user_move == 'Volume-Down':
        p.press('down')

    elif user_move == 'Volume-Up':
        p.press('up')

    

model = load_model(os.path.join('models','recognise.h5'))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame,(1450,900),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    else:
        continue

    cv2.rectangle(frame, (50, 100), (450, 500), (255, 0, 0), 2)

    # Extract the Region of Interest
    roi = frame[100:500, 50:450]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # predict the move made
    pred = model.predict(np.expand_dims(img/255,0))
    move_code = np.argmax(pred[0])
    user_move_name = class_map[move_code]

    font = cv2.FONT_HERSHEY_SIMPLEX

    if user_move_name == 'None':
        cv2.putText(frame,'Nothing to Recognise',(50, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        control(user_move_name)
        cv2.putText(frame,user_move_name,(50, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)



    cv2.imshow("Hand Sign Recognition", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
