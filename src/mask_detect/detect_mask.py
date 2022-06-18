import numpy as np
import cv2

def mask_image(img,model):
    prototxtPath = "./src/mask_detect/face_detector/deploy.prototxt"
    weightsPath = "./src/mask_detect/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    image = img
    #print(type(image),image)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    box_list = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box_list.append(box)
    return box_list
