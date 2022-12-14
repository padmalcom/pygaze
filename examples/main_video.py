import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pygaze import PyGaze
import cv2

pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")

v = cv2.VideoCapture(0)
while True:
	ret, frame = v.read()
	if ret:
		gaze_result = pg.predict(frame)
		for face in gaze_result:
			frame = pg.render(frame, face, draw_face_bbox=True, draw_face_landmarks=False, draw_3dface_model=False,draw_head_pose=False, draw_gaze_vector=True)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
v.release()
cv2.destroyAllWindows()		

