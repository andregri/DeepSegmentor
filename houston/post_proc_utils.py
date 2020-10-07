from visual_utils import *


def intBaricenter(points):
    y_mean = np.mean(points[0,:])
    x_mean = np.mean(points[1,:])
    return (int(y_mean), int(x_mean))


# Find where a road starts from the border of the image
def findRoadStartingPoints(matrix):
    assert len(matrix.shape) == 2
    start_points = []

    for row in [0, matrix.shape[0]-1]: # top and bottom border
        for col in range(0, matrix.shape[1]):
            if(matrix[row,col]==255):
                start_points.append((row,col))

    for col in [0, matrix.shape[1]-1]: # left and right border
        for row in range(0, matrix.shape[0]):
            if(matrix[row,col]==255):
                start_points.append((row,col))

    return start_points


def adjacent_edges(matrix, point):
    assert len(matrix.shape) == 2

    neighbours = [
        (point[0]-1, point[1]-1),
        (point[0]-1, point[1]  ),
        (point[0]-1, point[1]+1),
        (point[0],   point[1]+1),
        (point[0]+1, point[1]+1),
        (point[0]+1, point[1]  ),
        (point[0]+1, point[1]-1),
        (point[0]  , point[1]-1)
    ]
    edges = []
    for p in neighbours:
        if 0 <= p[0] < matrix.shape[0] and 0 <= p[1] < matrix.shape[1]:
            if matrix[p] != 0:
                edges.append(p)
    
    intersection = False
    if len(edges) == 3:
        intersection = True

    return edges, intersection


# DFS algorithm to find all road instances
def DFS(matrix, start_points):
    assert len(matrix.shape) == 2

    discovered = np.zeros(matrix.shape, dtype=np.uint8)
    road_instance = 1

    # DFS algorithm
    for v in start_points:
        S = []
        S.append(v)
        while len(S) != 0:
            v = S.pop()
            edges, intersect = adjacent_edges(matrix, v)
            if intersect:
                road_instance += 1
            if discovered[v] == 0:
                discovered[v] = road_instance
                for w in edges: 
                    S.append(w)
            #cv2.imshow('DFS', addCircles(thick_gt_bgr, [v]))
            #cv2.waitKey(5)
        road_instance += 1
    
    return discovered