def linear_move(point, scale):
    point *= scale
    assert(point[0] <= 200)
    assert(point[0] <= 200)
    assert(point[0] >= 0)
    assert(point[0] >= 0)
    return 'G01 X'+str(point[0])+' Y'+str(point[1])+' F1000'

def penup():
    return 'G01 Z1 F1000'

def pendown():
    return 'G01 Z-1 F1000'

def polygon_move(polygon, scale=1.0):
    ret = []
    ret.append(linear_move(polygon[0], scale))
    ret.append(pendown())
    for point in polygon[1:]:
        ret.append(linear_move(point, scale))
    return ret

def polygons_move(polygons, scale=1.0):
    ret = []
    for polygon in polygons:
        ret.append(penup())
        ret += polygon_move(polygon, scale)
    ret += 'G01 X0 Y0 F1000'
    return ret
