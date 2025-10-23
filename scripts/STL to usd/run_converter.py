"""This script calls a utility function to convert an STL mesh file
into USD format using specified parameters.
"""
from convert_mesh import mesh_converter

"""Possible inputs for collision_approximation: 

    "convexDecomposition"   (dynamic and complex shape) - composes multiple convex shapes to approximate the mesh shape 
    "convexHull"            (dynamic and simple shape) - uses smallest convex hull for collision approximation   
    "triangleMesh"          (only for static ) - use triangle mesh for collision approximation (most accurate, but computationally expensive) 
    "meshSimplification"    (trade-off between accuracy and performance) - simplified mesh for collision approximation
    "sdf"                   (good for soft bodies) - signed distance field for collision approximation
    "boundingCube"          (dynamic and low precision) - uses bounding cube for collision approximation
    "boundingSphere"        (dynamic and low precision) - uses bounding sphere for collision approximation
    "none" 
"""

input = "/home/ernstg/Downloads/Yoda.stl" #input file path of the STL file
output = "/home/ernstg/Desktop/Yoda USD.usd" #output file path of the USD file
mass = 1.0 #add mass to make it a rigid body, else None
collision_approximation = "convexDecomposition" #collision approximation method




mesh_converter(
    input_path=input,
    output_path=output,
    make_instanceable=False,
    collision_approximation=collision_approximation,
    mass=mass
)

