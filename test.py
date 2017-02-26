import pickle
from svg_slice import svg_slice
import matplotlib.pyplot as plt

polygons = pickle.load(open('polygons.pkl', 'rb'))

Ox, Oy = 30, 30

x_rng, y_rng = 100, 100

theta = 0.5

new_polygons = svg_slice(polygons, Ox, Oy, x_rng, y_rng, theta)

from polygon_gen import scatter_my_ass

#scatter_my_ass(new_polygons)

from gcode_gen import polygons_move

print '\n'.join(polygons_move(new_polygons))
