# pygaze
pygaze is a wrapper for the outstanding work of [pytorch_mpiigaze_demo](https://github.com/hysts/pytorch_mpiigaze_demo) and [eth-xgaze](https://ait.ethz.ch/projects/2020/ETH-XGaze/) to provide an api to estimate the gaze of humans.

![Gaze](img/gaze.jpg)

<a href="https://www.buymeacoffee.com/padmalcom" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## API
- First create an instance of *PyGaze*.
- Call *predict* on the instance to get a list of faces in the image and a gaze vector.

## Example usage
```python
from pygaze import PyGaze
import cv2

image = cv2.imread("jonas.jpg")
pg = PyGaze()
gaze_result = pg.predict(image)
for face in gaze_result:
    print(f"Face bounding box: {face.bbox}")
    pitch, yaw, roll = face.get_head_angles()
    g_pitch, g_yaw = face.get_gaze_angles()
    print(f"Face angles: pitch={pitch}, yaw={yaw}, roll={roll}.")
    print(f"Distance to camera: {face.distance}")
    print(f"Gaze angles: pitch={g_pitch}, yaw={g_yaw}")
	print(f"Gaze vector: {face.gaze_vector}")
```

## Todo
- Add a flag to *predict()* to pass already detected faces in an image.
- Add a function *render(img, face)* to draw the predicted gaze to an image.
- Add examples for realtime and image prediction.