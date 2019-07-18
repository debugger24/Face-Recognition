from embedding import getRep
import cv2

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    embedding = getRep(frame)
    print (embedding)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
