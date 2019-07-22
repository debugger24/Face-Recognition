from embedding import getRep
from predict import getPrediction
import cv2

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    embedding = getRep(frame)
    if (embedding is not None):
        print ('Predicted Person: ', getPrediction(embedding))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
