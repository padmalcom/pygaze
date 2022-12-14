import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pygaze import PyGaze
import cv2

image = cv2.imread("jonas.jpg")
pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")
gaze_result = pg.predict(image)
for face in gaze_result:
    print(face)
    img = pg.render(image, face)
    cv2.imshow("Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()