from pygaze import PyGaze
import cv2

image = cv2.imread("jonas.jpg")
pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")
print(pg.predict(image))