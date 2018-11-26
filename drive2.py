import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
from model2 import build_model
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask

from io import BytesIO

from keras.models import load_model
from keras.models import Model
from keras.models import Sequential


import utils2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None




@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        speed= float(data["speed"])
        throttle= float(data["throttle"])


        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = utils2.preprocess(image)
            image = np.array([image])

            [[steering_angle,throttle,brake]]= model.predict(image, batch_size=1)

            steering_angle= float(steering_angle)
            throttle = float(throttle)/3
            brake = float(brake)


            print('{} {} {}'.format(steering_angle, throttle, brake))
            send_control(steering_angle, throttle,brake)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0,0,0)


def send_control(steering_angle, throttle,brake):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),

            'throttle': throttle.__str__(),
            'brake': brake.__str__()

        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    model=load_model('model-003i.h5')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.Middleware(sio, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
