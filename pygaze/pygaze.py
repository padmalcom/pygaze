from omegaconf import OmegaConf
from loguru import logger
import os
import pathlib
import torch
import cv2
from gaze_estimator import GazeEstimator

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
		
		# initialize
		self.gaze_estimator = GazeEstimator(self.config)
				
	def predict(self, img):
		results = []
		if img is None:
			logger.warning("Invalid image.")
			return results
			
		undistorted = cv2.undistort(img, self.gaze_estimator.camera.camera_matrix, self.gaze_estimator.camera.dist_coefficients)
		#cv2.imshow("undistorted", undistorted)
		#cv2.waitKey(0) # waits until a key is pressed
		#cv2.destroyAllWindows()
		
		faces = self.gaze_estimator.detect_faces(undistorted)
		logger.debug("Found {} face(s).", len(faces))

		for face in faces:
			print(face)
			self.gaze_estimator.estimate_gaze(undistorted, face)
			#self._draw_face_bbox(face)
			#self._draw_head_pose(face)
			#self._draw_landmarks(face)
			#self._draw_face_template_model(face)
			#self._draw_gaze_vector(face)
			#self._display_normalized_image(face)
		return faces
			