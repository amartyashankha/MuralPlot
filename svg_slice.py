import numpy as np
from svg.path import parse_path

def svg_slice(polygons, Ox, Oy, x_rng, y_rng, theta):
    
    def transform(point, origin, theta):
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return R.dot(np.array(point-origin))
    
    def check_bounds(point, max_coord):
        return 0<point[0]<max_coord[0] and 0<point[1]<max_coord[1] 
    
    def check_dist(p1, p2):
        return np.linalg.norm(p1-p2) < 1.0
    
    def bound_box(polygons, origin, rng, theta):
        new_polygons = []
        for polygon in polygons:
            new_polygon = []
            for point in polygon+[[float('inf'), 0]]:
                p = transform(point, origin, theta)
                if check_bounds(p, rng) and (len(new_polygon) == 0 or check_dist(p, new_polygon[-1])):
                    new_polygon.append(p)
                elif len(new_polygon) > 0:
                    new_polygons.append(new_polygon)
                    new_polygon = []
        return new_polygons
    
    origin = np.array([Ox, Oy])
    rng = np.array([x_rng, y_rng])
    
    return bound_box(polygons, origin, rng, theta)
