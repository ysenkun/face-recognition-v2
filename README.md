# Face Recognition with FaceNet

This repository is a face recognition system based on FaceNet, and major rewrite of the [previous one](https://github.com/ysenkun/face_recognition).
### :raising_hand: Reference:
1. https://github.com/davidsandberg/facenet

## Seting Up Environment
The Python environment is python==3.8
```bash
$ git clone https://github.com/ysenkun/face_recognition_v2.git
```
```bash
$ conda create -n facenet python==3.8
$ conda activate facenet
```
```bash
$ pip3 install -r requirements.txt
```

### Pre-trained Models(FaceNet)
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

* NOTE: If you use any of the models, please do not forget to give proper credits to those providing the training datasets as well.
Where to save the model:
```bash
$ mkdir src/facenet_model
```
Download the vggface2 model and rename the folder to vggface2
```bash
$ unzip ~/Downloads/20180402-114759.zip
```
```bash
$ mv 20180402-114759 facenet_model/vggface2
```

### Pre-trained Models (For face detection)
After downloading [mask_detector.model](https://drive.google.com/file/d/1DdaF3eRnlbv2ssvsJhHqlGQTnlhqK2wi/view?usp=sharing), save it in the following directory
```bash
$ mv ~/Downloads/mask_detector.model src/mask_detect
```
### :raising_hand: Reference:
1. https://github.com/chandrikadeb7/Face-Mask-Detection

## Modifying Shell Script
Modify _{YOUR_APPROPRIATE_PATH}/facenet/bin/python_ to appropriate file path.  
You need to modify 1 location in run.sh

## Run
The following shell script would first prompt the user to define how many faces to be registered. Then it would take a photo for each user. Finally, it would make a database for the detection.
```bash
$ bash run.sh create
```

### Face Recognition
The following command would perform a face recognition based on the faces registered.
```bash
$ bash run.sh camera
```
![sample](https://user-images.githubusercontent.com/82140392/163969654-e555e41f-aa25-42d6-9c9e-c1fc5bc79a09.gif)
