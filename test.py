import pickle
from svg_slice import svg_slice
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tracking import Tracker

polygons = pickle.load(open('polygons.pkl', 'rb'))
''''
M = np.array([[  6.65711273e-02,   5.47559773e+00,  -1.89238360e+02],
       [ -5.39128636e+00,   1.01174157e-01,   1.79414588e+03],
       [  6.35660769e-05,   8.09843393e-05,   1.00000000e+00]])

import subprocess

#bashCommand = "streamer -c /dev/video1 -b 16 -o image.jpeg"

#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

#output, error = process.communicate()

fname = '/home/shankha/MuralPlot/left.jpeg'
img = cv2.imread(fname)
im_out = cv2.warpPerspective(img, M, (1000,1000))

plt.imsave(im_out, 'scaled.jpeg')
'''
Ox, Oy = 0, 0

x_rng, y_rng = 2000, 2000

theta = 0.0

new_polygons = svg_slice(polygons, Ox, Oy, x_rng, y_rng, theta)

from polygon_gen import scatter_my_ass

#scatter_my_ass(new_polygons)
#plt.show()

from gcode_gen import polygons_move

print '\n'.join(polygons_move(new_polygons, scale=0.12))
