# Distance Measurement

Webcam-based distance estimation using the pinhole camera model: `distance = W * f / w`, where `W` is an assumed real-world width of the object, `f` is an assumed focal length in pixels, and `w` is the object's width in the image. `code.py` runs YOLOv5s on each frame and applies this formula to every detection. `Face Distance.ipynb` holds several variants, including a face-mesh version and contour-based versions that need no neural network.

## Features

- YOLOv5s object detection with distance labels on the webcam feed (`code.py`)
- Face-mesh distance using `cvzone.FaceMeshDetector` (eye landmarks 145 and 374), in the notebook and the commented block at the top of `code.py`
- Per-class real widths in the notebook (person 50 cm, car 180 cm, dog 50 cm, bicycle 50 cm, default 100 cm)
- Canny-based contour detection for a single object, with and without YOLO
- Helper functions in `utlis.py`: contour detection, corner reordering, perspective warp and point distance

## Project structure

- `code.py`: webcam script. Loads YOLOv5s through `torch.hub`, uses a placeholder width of 6.3 cm and focal length 840 px.
- `Face Distance.ipynb`: experiments, in order: face mesh, YOLOv5 on webcam, YOLOv5 with per-class widths on `sample3.jpg`, Canny contour on `sample6.jpg`, and Canny contour on webcam.
- `utlis.py`: OpenCV helper functions (`getContours`, `reorder`, `warpImg`, `findDis`). `code.py` does not import it.
- `yolov5s.pt`, `yolov8n.pt`: weight files. `code.py` does not load these from disk; it fetches YOLOv5s through `torch.hub`.

## Requirements

Imports used: `cv2`, `cvzone`, `numpy`, `torch`. The face-mesh variant also needs `mediapipe`, which `cvzone` uses.

```
pip install opencv-python cvzone numpy torch mediapipe
```

## Installation

```
git clone https://github.com/Yaldram69/Distance-Measurement.git
cd Distance-Measurement
pip install opencv-python cvzone numpy torch mediapipe
```

`torch.hub.load('ultralytics/yolov5', 'yolov5s')` needs internet access on first run and downloads the YOLOv5 repository and weights.

## Usage

```
python code.py
```

Uses the default webcam (index 0). No arguments. The loop has no quit key, so close the window or press Ctrl+C to stop.

The notebook `Face Distance.ipynb` expects `sample3.jpg` and `sample6.jpg` in the working folder. These images are not in the repository, so replace the filenames with your own images.

## Notes

- Distance is only as good as `W` and `f`. The values 6.3 cm and 840 px are placeholders; calibrate the focal length for your camera and set `W` to the real width of the object.
- In `code.py`, the same `W` is applied to every detected class, so labels are meaningful only for objects of that width.
- The distance is in the unit of `W` (cm in the code's comments).
- `utlis.py` is a helper library and is not used by the scripts in this repository.
- Results depend on the camera being roughly perpendicular to the object and on the object's apparent width being accurate.

## Pretrained YOLO weights (not included)

The following off-the-shelf Ultralytics checkpoints were removed from this repository because they are publicly downloadable. The code loads them by name, and Ultralytics downloads them automatically on first use:

- `yolov5s.pt`
- `yolov8n.pt`

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # downloaded automatically if not present
```

Or download them manually from https://github.com/ultralytics/assets/releases
