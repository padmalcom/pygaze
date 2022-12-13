import bz2
import logging
import operator
import pathlib
import tempfile
from loguru import logger
import cv2
import torch.hub
import yaml
from omegaconf import DictConfig

from common.face_model import FaceModel
from common.face_model_68 import FaceModel68
from common.face_model_mediapipe import FaceModelMediaPipe


def get_3d_face_model(config: DictConfig) -> FaceModel:
	return FaceModelMediaPipe()
