import pickle
from svg_slice import svg_slice
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tracking import Tracker

polygons = pickle.load(open('polygons.pkl', 'rb'))

M = np.array([[  1.33204016e-02,   1.09428485e+00,  -3.86193966e+01],
              [ -1.08063537e+00,   1.69380537e-02,   3.17952045e+02],
              [  6.25917503e-05,   7.89107784e-05,   1.00000000e+00]])

import subprocess

bashCommand = "streamer -c /dev/video1 -b 16 -o image.jpeg"

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

output, error = process.communicate()

fname = '/home/shankha/MuralPlot/image.jpeg'
img = cv2.imread(fname)
im_out = cv2.warpPerspective(img, M, (1000,1000))


Ox, Oy = 0, 0

x_rng, y_rng = 40, 40

theta = 0.0

new_polygons = svg_slice(polygons, Ox, Oy, x_rng, y_rng, theta)

from polygon_gen import scatter_my_ass

#scatter_my_ass(new_polygons)
#plt.show()

from gcode_gen import polygons_move

print '\n'.join(polygons_move(new_polygons, scale=1.0))
