import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pygaze import PyGaze
import cv2
from loguru import logger

image = cv2.imread("jonas.jpg")
pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")
gaze_result = pg.predict(image)
logger.debug("Found {} face(s).", len(gaze_result))
for face in gaze_result:
    print(f"Face bounding box: {face.bbox}")
    pitch, yaw, roll = face.get_head_angles()
    g_pitch, g_yaw = face.get_gaze_angles()
    print(f"Face angles: pitch={pitch}, yaw={yaw}, roll={roll}.")
    print(f"Distance to camera: {face.distance}")
    print(f"Gaze angles: pitch={g_pitch}, yaw={g_yaw}")
    print(f"Gaze vector: {face.gaze_vector}")
    img = pg.render(image, face, draw_face_bbox=True, draw_face_landmarks=False, draw_3dface_model=False,draw_head_pose=False, draw_gaze_vector=True)
    cv2.imshow("Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()