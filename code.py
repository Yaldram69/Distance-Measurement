# import cv2
# import cvzone
# from cvzone.FaceMeshModule import FaceMeshDetector
# import numpy as np
# import mediapipe as mp
#
# cap = cv2.VideoCapture(0)
# detector = FaceMeshDetector(maxFaces=1)
#
# sen = 25  # more is less
#
# while True:
#     success, img = cap.read()
#     img, faces = detector.findFaceMesh(img, draw=False)
#
#     if faces:
#         face = faces[0]
#         pointLeft = face[145]
#         pointRight = face[374]
#         w, _ = detector.findDistance(pointLeft, pointRight)
#         W = 6.3
#
#         # Finding distance
#         f = 840
#         d = (W * f) / w
#         print(d)
#
#         cvzone.putTextRect(img, f'Depth: {int(d)}cm',
#                            (face[10][0] - 100, face[10][1] - 50),
#                            scale=2)
#
#     cv2.imshow("Webcam Feed", img)
#     cv2.waitKey(1)





import cv2
import cvzone
import numpy as np
import torch  # YOLOv5 requires PyTorch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model, you can change 'yolov5s' to 'yolov5m' or 'yolov5l' for larger models

cap = cv2.VideoCapture(0)

sen = 25  # more is less

while True:
    success, img = cap.read()
    results = model(img)  # Run YOLOv5 detection

    # Process results
    img = np.squeeze(results.render())  # Get the rendered image (with bounding boxes and labels)

    # If any objects are detected
    if results.xywh[0].size(0) > 0:
        for det in results.xywh[0]:
            x, y, w, h, conf, cls = det  # Get the bounding box (x, y, width, height), confidence, and class
            label = results.names[int(cls)]  # Get the object label

            # Assuming the real-world width of the object (W) and focal length (f) are known
            # For now, using a placeholder width of 6.3 cm for demonstration
            W = 6.3  # Real-world width of the object (in cm)
            f = 840  # Focal length (you can adjust based on your camera specs)
            d = (W * f) / w  # Calculate the distance to the object

            # Display the distance on the image
            cvzone.putTextRect(img, f'Distance: {int(d)}cm', (int(x), int(y) - 10), scale=2)

    cv2.imshow("Webcam Feed", img)
    cv2.waitKey(1)
