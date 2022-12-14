from omegaconf import OmegaConf
from loguru import logger
import os
import pathlib
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .utils import get_3d_face_model
from .gaze_estimator import GazeEstimator

class GazeResult:
	def __init__(self):
		pass

class PyGaze:

	def __download_model__(self, target_dir):
		logger.debug('Downloading model to {}...', target_dir)
		output_dir = pathlib.Path(target_dir).expanduser()
		output_dir.mkdir(exist_ok=True, parents=True)
		output_path = os.path.join(output_dir, 'eth-xgaze_resnet18.pth')
		if not os.path.exists(output_path):
			logger.debug('Download the pretrained model...')
			torch.hub.download_url_to_file(
				'https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth',
			output_path)
		else:
			logger.debug('The pretrained model {} already exists.', output_path)
		return output_path

	def __init__(self, device="cpu", model_path = "~/.ptgaze/models"):

		self.config = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/eth-xgaze.yaml'))
		self.config.PACKAGE_ROOT = pathlib.Path(__file__).parent.resolve().as_posix()
		self.config.device = device
		
		# check the model path and download the model
		self.config.gaze_estimator.checkpoint = os.path.abspath(model_path)
		if os.path.isfile(self.config.gaze_estimator.checkpoint):
			logger.warning("{} is a file path but a directory is required. Removing the filename...", self.config.gaze_estimator.checkpoint)
			self.config.gaze_estimator.checkpoint = os.path.dirname(self.config.gaze_estimator.checkpoint)
		self.config.gaze_estimator.checkpoint = self.__download_model__(self.config.gaze_estimator.checkpoint)

		# for visualization only
		self.head_pose_axis_length = 0.05
		self.gaze_visualization_length = 0.05
		self.AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
		self.face_3d_model = get_3d_face_model(self.config)


		# initialize
		self.gaze_estimator = GazeEstimator(self.config)
				
	def predict(self, img):
		results = []
		if img is None:
			logger.warning("Invalid image.")
			return results
			
		undistorted = cv2.undistort(img, self.gaze_estimator.camera.camera_matrix, self.gaze_estimator.camera.dist_coefficients)		
		faces = self.gaze_estimator.detect_faces(undistorted)
		for face in faces:
			self.gaze_estimator.estimate_gaze(undistorted, face)
		return faces

	def render(self, img, face, draw_face_bbox=True, draw_face_landmarks=True, draw_3dface_model=True,draw_head_pose=True, draw_gaze_vector=True):
		image = img.copy()
		size = 1
		color = (0, 255, 0)

		if draw_face_bbox:
			bbox = np.round(face.bbox).astype(np.int).tolist()
			image = cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), color, size)

		if draw_face_landmarks:
			for pt in face.landmarks:
				pt = tuple(np.round(pt).astype(np.int).tolist())
				image = cv2.circle(image, pt, size, color, cv2.FILLED)

		if draw_3dface_model:
			points2d = self.gaze_estimator.camera.project_points(face.model3d)
			for pt in points2d:
				pt = tuple(np.round(pt).astype(np.int).tolist())
				image = cv2.circle(image, pt, size, color, cv2.FILLED)

		if draw_head_pose:
			axes3d = np.eye(3, dtype=np.float) @ Rotation.from_euler('XYZ', [0, np.pi, 0]).as_matrix()
			axes3d = axes3d * self.head_pose_axis_length
			axes2d = self.gaze_estimator.camera.project_points(axes3d, face.head_pose_rot.as_rotvec(), face.head_position)
			center = face.landmarks[self.face_3d_model.NOSE_INDEX]
			center = tuple(np.round(center).astype(np.int).tolist())
			for pt, color in zip(axes2d, self.AXIS_COLORS):
				pt = tuple(np.round(pt).astype(np.int).tolist())
				image = cv2.line(image, center, pt, color, 2, cv2.LINE_AA)

		if draw_gaze_vector:
			start = face.center
			end = face.center + self.gaze_visualization_length * face.gaze_vector
			points3d = np.vstack([start, end])
			points2d = self.gaze_estimator.camera.project_points(points3d)
			pt0 = tuple(np.round(points2d[0]).astype(np.int).tolist())#(points2d[0])
			pt1 = tuple(np.round(points2d[1]).astype(np.int).tolist())#(points2d[1])
			image = cv2.line(image, pt0, pt1, color, 1, cv2.LINE_AA)
			
		return image
			