# pygaze
pygaze is a library based on https://github.com/hysts/pytorch_mpiigaze_demo and eth-xgaze to estimate the gaze of humans.

<a href="https://www.buymeacoffee.com/padmalcom" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## API
- First create an instance of *PyGaze*.
- Call *predict* on the instance to get a list of faces in the image and a gaze vector.

## Example usage

```python
from pygaze import PyGaze
import cv2

image = cv2.imread("jonas.jpg")
pg = PyGaze(model_path="models/eth-xgaze_resnet18.pth")
print(pg.predict(image))
```

## Todo
- Add a flag to *predict* to pass already detected faces in an image.