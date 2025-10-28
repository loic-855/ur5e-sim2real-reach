"""This script calls a utility function to convert an STL mesh file
into USD format using specified parameters.
"""
import os
from pathlib import Path
from convert_mesh_to_usd import mesh_converter

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
base_dir = Path(__file__).resolve().parent

root_dir = base_dir.parents[1]

input_folder = root_dir / "STL_files" #input folder path
output_folder = root_dir / "USD_files" #output folder path

mass = 1.0 #add mass to make it a rigid body, else None
collision_approximation = "convexDecomposition" #collision approximation method

stl_files = list(input_folder.glob("*.stl"))
if not stl_files:
    print("⚠️ No STL-files found.")
else:
    print(f"🔍 {len(stl_files)} STL-files found:\n")

#loop through all STL files in the input folder
for stl_path in stl_files:
    base_name = stl_path.stem
    usd_path = output_folder / f"{base_name}.usd"

    print(f"➡️ Convert {stl_path.name} → {usd_path.name}")

    try:
        mesh_converter(
            input_path=str(stl_path),
            output_path=str(usd_path),
            make_instanceable=False,
            collision_approximation=collision_approximation,
            mass=mass
        )
        print(f"✅ {usd_path.name} succesfull!.\n")

    except Exception as e:
        print(f"❌ Error {stl_path.name}: {e}\n")