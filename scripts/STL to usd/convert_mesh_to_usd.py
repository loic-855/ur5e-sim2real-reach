"""This script converts a mesh file (e.g., STL, OBJ) into USD format and adds optional
rigid body and collision properties based on user-defined parameters."""

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os

import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def mesh_converter(
    input_path: str,
    output_path: str,
    make_instanceable: bool = False,
    collision_approximation: str = "convexDecomposition",
    mass: float | None = None,
):
    # --- Validierung ---

    # Mass & Rigid Body
    if mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    # Collision settings
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=collision_approximation != "none"
    )

    collision_approximation_map = {
        "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
        "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
        "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
        "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
        "sdf": schemas_cfg.SDFMeshPropertiesCfg,
        "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
        "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
        "none": None,
    }

    cfg_class = collision_approximation_map.get(collision_approximation)
    collision_cfg = cfg_class() if cfg_class is not None else None

    # --- Mesh Converter Config ---

    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=input_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(output_path),
        usd_file_name=os.path.basename(output_path),
        make_instanceable=make_instanceable,
        mesh_collision_props=collision_cfg,
    )

    # --- Konvertierung ausführen ---
    mesh_converter = MeshConverter(mesh_converter_cfg)
    print(f"✅ Converted mesh '{input_path}' to USD format")
    print(f"✅ USD file generated at: {mesh_converter.usd_path}")


#   simulation_app.close()
