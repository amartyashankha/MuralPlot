import xml.etree.ElementTree as ET
import numpy as np
from svg.path import parse_path
import matplotlib.pyplot as plt

def polygon_gen(fname, resolution=2, error=1e-6):

    tree = ET.parse(fname)
    root = tree.getroot()
    svg = list(root.findall('*'))[-1]
    
    polygons = []
    
    for p in list(svg.findall('{http://www.w3.org/2000/svg}path')):
        path = parse_path(p.get('d'))
        num_points = int(path.length(error=error)*resolution)
        print num_points
        points = [path.point(i*1.0/num_points, error=error) for i in range(num_points)]
        points = [np.array([point.real, point.imag]) for point in points]
        polygons.append(points)
    
    return polygons


def scatter_my_ass(polygons):
    x_values = []
    y_values = []

    for polygon in polygons:
        x_values += [point[0] for point in polygon]
        y_values += [point[1] for point in polygon]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x_values, y_values, s=0.1)

fname = '/home/shankha/MuralPlot/svg/brain.svg'

polygons = polygon_gen(fname)

import pickle

print len(polygons)

pickle.dump(polygons, open('polygons.pkl', 'wb'))
