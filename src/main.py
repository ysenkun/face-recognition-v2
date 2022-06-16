from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import load_model
from skimage.transform import resize
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import sqlite3
import cv2
import asyncio
import time
from mask_detect import detect_mask

class main():
    def tensorflow_activate():
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with tf.Session() as sess:

                # Load the model for FaceNet
                facenet.load_model(args.model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                if args.action == 'create':
                    create_db.visualization(sess, images_placeholder, embeddings, phase_train_placeholder)
                else:
                    recognition.camera(sess, images_placeholder, embeddings, phase_train_placeholder)

    #Change function name to "img_resize" in the future
    def _img_resize(frames):
        image_size = args.image_size
        margin = args.margin
        
        img_list = []
        images_list = []
        box_list = []
        img_list_db = []
        for k, img in enumerate(frames):
            img_size = np.asarray(img.shape)[0:2]
            #Use any face detection library
            box = detect_mask.mask_image(img, model)
            box_list += box
            images_list = []
            for det in box:
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                try:
                    aligned = resize(cropped, (image_size, image_size))
                except:
                    continue
                prewhitened = facenet.prewhiten(aligned)
                img_list_db.append(prewhitened)
                img_list.append(prewhitened)
                images = np.stack(img_list)
                images_list.append(images)
                img_list.clear()
        if args.action == 'create':
            images_db = np.stack(img_list_db)
        else:
            images_db = 'none'
        return images_list,box_list, images_db

        
class create_db():

    def camera():
        cap = cv2.VideoCapture(0)
        finish_num = int(input("\n Input the number of photos \n"))
        n = 1
        frames = []
        names = []
        for i in range(finish_num):
            user_name = input("\n Input your name \n")
            print("Press A for the photo shoot")
            while True:
                ret, frame = cap.read()
                cv2.imshow('cap', frame)
                key = cv2.waitKey(33)

                if key == ord('a'):
                    num = format(n,'03')
                    frames.append(frame)
                    #cv2.imwrite('./src/data/images/{}{}.jpg'.format(user_name,num), frame)
                    names.append(user_name)
                    n += 1
                    break
                elif key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        print("That's a wrap \n Creates a DB")
        return frames, names

    def visualization(sess, images_placeholder, embeddings, phase_train_placeholder):
        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
        emb = sess.run(embeddings, feed_dict=feed_dict)        

        nrof_images = len(images)
        for i in range(nrof_images):
            name = names[i]
            print(name)
            data = list(emb[i,:])
            create_db.insert_db(name, data)

        print('Distance matrix')
        print('    ', end='')
        for i in range(nrof_images):
            name = names[i]
            print('    %1d     ' % i, end='')
        print('')
        for i in range(nrof_images):
            print('%1d  ' % i, end='')
            for j in range(nrof_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                print('  %1.4f  ' % dist, end='')
            print('')        

    def insert_db(name, data):
        dbname = './src/register.db'
        conn = sqlite3.connect(dbname)
        cur = conn.cursor()

        try:
            cur.execute('CREATE TABLE persons(name STRING, data REAL)')
            cur.execute('INSERT INTO persons(name,data) values("{}","{}")'.format(name,data))
        except:
            cur.execute('INSERT INTO persons(name,data) values("{}","{}")'.format(name,data))

        conn.commit()
        cur.close
        conn.close()

class recognition():

    def __init__(self):
        self.dbname = './src/register.db'
        self.conn = sqlite3.connect(self.dbname)
        self.cur = self.conn.cursor()
        self.select_sql = 'SELECT * FROM persons'
    
    def camera(self, sess, images_placeholder, embeddings, phase_train_placeholder):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            images_list, box_list, none = main._img_resize([frame])
            # Run forward pass to calculate embeddings
            for num,images in enumerate(images_list):
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                name_detect = recognition.face_detect(emb)
                # cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame,name_detect,(int(box_list[num][0]+30), int(box_list[num][1])-30),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('cap',frame)
            if cv2.waitKey(1) == ord('q'):
                break

    def face_detect(self,emb):
        for row in self.cur.execute(self.select_sql):
            name = row[0]
            data = np.array(eval(row[1]))
            dis = np.sqrt(np.sum(np.square(np.subtract(data,emb[0,:]))))
            if dis < 0.65:
                return name
        return 'unknown'

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--action', type=str,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    model = load_model("./src/mask_detect/mask_detector.model")
    args = parse_arguments(sys.argv[1:])

    #Create DB
    if args.action == 'create':
        frames, names = create_db.camera()
        images, box_list, images_db = main._img_resize(frames)
        images = images_db
        main.tensorflow_activate()

    #Face recognition
    elif args.action == 'camera':
        recognition = recognition()
        main.tensorflow_activate()
