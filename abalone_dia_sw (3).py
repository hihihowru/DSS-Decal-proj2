import csv
import math as m
import random

filename = 'abalone.data.txt'

with open(filename) as f:
    """Loads data into a 2D python list"""
    reader = csv.reader(f)
    data = list(reader)

diameter = [point[2] for point in data]
weight = [point[5] for point in data]
points = [(float(p[2]), float(p[5])) for p in data]

def mean(s):
    """Returns the mean value of a python list"""
    assert s
    return sum(s)/len(s)

def distance(a, b):
    """Distance formula"""
    return m.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def initial_centroids(points, n):
    """Picks n random points to be the inital centroids"""
    return random.sample(points, n)

def group(points, centroids):
    """Maps each point to the closest centroid, storing
       it in a dictionary with the centroids as keys"""
    dict = {}
    for c in centroids:
        dict[c] = []
    for p in points:
        closest = min(centroids, key=lambda x: distance(x, p))
        dict[closest].append(p)
    return dict

def find_centroid(cluster):
    """Computes the centroid of a cluster"""
    ave_diameter = mean([point[0] for point in cluster])
    ave_weight = mean([point[1] for point in cluster])
    return (ave_diameter, ave_weight)

def k_means(points, k, max_updates=1000):
    """
    Runs a k-means algorithm which groups a dataset, points,
    into k number of points. The algorith will run max_updates
    number of times or until the correct centroids are found.
    """
    centroids = initial_centroids(points, k)
    while max_updates:
        old_centroids = centroids
        cluster = group(points, centroids)
        centroids = [find_centroid(vals) for k, vals in cluster.items()]
        if centroids == old_centroids:
            break
    return centroids

centroids = k_means(points, 5)
print(centroids)
