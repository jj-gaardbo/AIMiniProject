#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import cv2

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:

    #  Wait for next request from client
    message = socket.recv()

    time.sleep(0.1) #Small delay to wait for the texture to be made

    f = open('image.jpg', 'wb')
    f.write(message)
    f.close()

    #path = r'..\UnityProject\AssetsSavedScreen.png'
    path = r'image.jpg'
    image = cv2.imread(path, 1)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    #  In the real world usage, you just need to replace time.sleep() with
    #  whatever work you want python to do.
    time.sleep(1)

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    socket.send(b"World")



