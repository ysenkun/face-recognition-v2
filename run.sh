#!/bin/sh
model='./src/facenet_model/vggface2/20180402-114759.pb'

#Update DB
if [[ ${1} = 'create' ]]; then
	rm ./register.db
fi

#run
/usr/local/anaconda3/envs/facenet/bin/python3 src/main.py ${model} --image_size 160 --margin 32 --gpu_memory_fraction 0 --action ${1}
