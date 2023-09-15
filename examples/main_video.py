import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pygaze import PyGaze, PyGazeRenderer
import cv2
import math

pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")
pgren = PyGazeRenderer()
v = cv2.VideoCapture(0)
min_x = 100.0
max_x = -100.0
min_y = 100.0
max_y = -100.0

while True:
	ret, frame = v.read()
	if ret:
		gaze_result = pg.predict(frame)
		for face in gaze_result:
			color = (0, 255, 0)
			if pg.look_at_camera(face):
				color = (255, 0, 0)
			pgren.render(
				frame,
				face,
				draw_face_bbox=True,
				draw_face_landmarks=False,
				draw_3dface_model=False,
				draw_head_pose=False,
				draw_gaze_vector=True,
				color = color
			)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
v.release()
cv2.destroyAllWindows()		

