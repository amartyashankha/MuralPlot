def linear_move(point):
    return 'G01 X'+str(point[0])+' Y'+str(point[1])

def penup():
    return 'G Z1'

def pendown():
    return 'G Z-1'

def polygon_move(polygon):
    ret = []
    ret.append(linear_move(polygon[0]))
    ret.append(pendown())
    for point in polygon[1:]:
        ret.append(linear_move(point))
    return ret

def polygons_move(polygons):
    ret = []
    for polygon in polygons:
        ret.append(penup())
        ret += polygon_move(polygon)
    return ret
