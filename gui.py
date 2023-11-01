from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from keras.models import load_model
from imutils import face_utils
import dlib

best_model = load_model('eye_model.h5')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

YAWN_THRESH = 20
COUNTER = 0
def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

model = FacialExpressionModel("model_a.json", "model_weights.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def detect_eyes(processed_eye):
    eye_prediction = best_model.predict(processed_eye)
    return eye_prediction

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)
        distance = lip_distance(shape)
        lip = shape[48:60]
        #cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        if (distance > YAWN_THRESH):
            cv2.putText(frame, "Mouth Open ", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Mouth Close ", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        eyes_roi = gray_frame[y:y + h, x:x + w]

        emotion = EMOTIONS_LIST[np.argmax(model.predict(face_roi[np.newaxis, :, :, np.newaxis]))]
        print("Predicted Emotion is " + emotion)

        eyes = eye_cascade.detectMultiScale(eyes_roi, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1)
            eye = eyes_roi[ey:ey + eh, ex:ex + ew]
            processed_eye = cv2.resize(eyes_roi, (224, 224))
            processed_eye = processed_eye / 255.0
            if processed_eye.shape[-1] != 3:
                processed_eye = np.stack((processed_eye,) * 3, axis=-1)
            processed_eye = np.expand_dims(processed_eye, axis=0)
            eye_prediction = detect_eyes(processed_eye)
            if eye_prediction > 0.5:
                eye_status = 'Open'
            else:
                eye_status = 'Closed'
            cv2.putText(frame, f'Eye: {eye_status}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
