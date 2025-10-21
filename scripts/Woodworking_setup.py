# Robot setup for IDEALAB at ETHZ by Pascal Aebersold
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script shows the basic setup of two UR5e robots on a table.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from pxr import Usd, UsdPhysics, UsdGeom, Sdf
import omni.usd
import math


#----------------------------Set Render to Performance----------------------------#
render_cfg = sim_utils.RenderCfg(rendering_mode="performance")

#------------------------------Import Woodworking Table---------------------------#
from pathlib import Path
# Resolve to the root of project
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
# Path to asset folder
IDEALAB_ASSET_DIR = PROJECT_ROOT_DIR / "Woodworking_Simulation" / "source" / "Woodworking_Simulation" / "Woodworking_Simulation" / "tasks" / "manager_based" / "woodworking_simulation" / "asset"
# Table Width = 1.2m
# Table Depth = 0.8m
# Table Height = 0.842m

#--------------------------------Import UR5e Robot--------------------------------#
UR5E_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd"),
    actuators={"arm_action": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

#----------------------------------Import Gripper---------------------------------#
GRIPPER_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{IDEALAB_ASSET_DIR}/onrobot_2fg7_expanded.usd"),
) 

#-----------------------------Creat Orgins for Robots-----------------------------#
def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=2, spacing=-1.12)

    #----------------------------------Origin1------------------------------------#
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    #-----------------------------------Table-------------------------------------#
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{IDEALAB_ASSET_DIR}/Assambly_Table.usd", scale=(0.001, 0.001, 0.001)
    )
    cfg.func("/World/Origin1/Table", cfg, translation=(0.0, 0.0, 0.0))

    #-------------------------------Robot1 in front-------------------------------#
    ur5e1_cfg = UR5E_CONFIG.replace(prim_path="/World/Origin1/Robot1")
    ur5e1_cfg.init_state.pos = (0.720, -0.08, 0.842)
    ur5e1_cfg.init_state.joint_pos = {"shoulder_pan_joint":1.57, "shoulder_lift_joint":-1.57,  "elbow_joint":-1.57, "wrist_1_joint":-1.57, "wrist_2_joint":1.57, "wrist_2_joint":1.57}
    ur5e1 = Articulation(cfg=ur5e1_cfg)

    #-----------------------------Robot2 in the back------------------------------#
    ur5e2_cfg = UR5E_CONFIG.replace(prim_path="/World/Origin2/Robot2")
    ur5e2_cfg.init_state.pos = (0.08, 0, 0.842)
    ur5e2_cfg.init_state.joint_pos = {"shoulder_pan_joint":-1.57, "shoulder_lift_joint":-1.57,  "elbow_joint":-1.57, "wrist_1_joint":-1.57, "wrist_2_joint":1.57, "wrist_2_joint":1.57}
    ur5e2 = Articulation(cfg=ur5e2_cfg)

    #------------------------------Gripper for Robot2-----------------------------#
    stage = omni.usd.get_context().get_stage()

    # flange prim path on the robot (adjust if your UR5e USD uses a different flange prim name)
    flange_path = Sdf.Path("/World/Origin2/Robot2/wrist_3_link")
    flange_prim = stage.GetPrimAtPath(flange_path)
    if not flange_prim.IsValid():
        raise RuntimeError(f"Flange prim not found at {flange_path}")

    # read flange world transform
    xform_cache = UsdGeom.XformCache()
    flange_xform = xform_cache.GetLocalToWorldTransform(flange_prim)  # Gf.Matrix4d
    flange_pos = flange_xform.ExtractTranslation()
    flange_rot = flange_xform.ExtractRotation()  # Gf.Rotation (axis+angle)
    axis = flange_rot.GetAxis()
    angle = flange_rot.GetAngle()
    s = math.sin(angle * 0.5)
    flange_orient = (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(math.cos(angle * 0.5)))

    # spawn gripper as sibling articulation root (avoid putting it under Robot2)
    gripper_root_path = "/World/Origin2/Robot2_Gripper"
    gripper_cfg = GRIPPER_CONFIG.replace(prim_path=gripper_root_path)
    gripper_cfg.actuators = {
        "gripper_action": ImplicitActuatorCfg(
            joint_names_expr=["left_finger_joint.*", "right_finger_joint.*"], damping=None, stiffness=None
        )
    }
    gripper_cfg.init_state.pos = (float(flange_pos[0]), float(flange_pos[1]), float(flange_pos[2]))
    gripper_cfg.init_state.orient = flange_orient
    gripper = Articulation(cfg=gripper_cfg)

    # helper: find first prim under root that has a UsdPhysics.RigidBody schema
    def find_rigid_body_prim(root_path: str) -> Sdf.Path | None:
        root = stage.GetPrimAtPath(Sdf.Path(root_path))
        if not root.IsValid():
            return None
        for p in root.GetAllChildren():  # direct children first
            if UsdPhysics.RigidBodyAPI.Get(stage, p.GetPath()):
                return p.GetPath()
        # recursive fallback
        def recurse(prim):
            for child in prim.GetChildren():
                if UsdPhysics.RigidBodyAPI.Get(stage, child.GetPath()):
                    return child.GetPath()
                res = recurse(child)
                if res:
                    return res
            return None
        return recurse(root)

    # find flange rigid body prim (body0) and gripper rigid body prim (body1)
    body0 = find_rigid_body_prim(str(flange_path)) or flange_path
    body1 = find_rigid_body_prim(gripper_root_path)
    if body1 is None:
        # If we cannot find a rigid-body under the gripper root, print tree to debug
        print("[WARN] could not find gripper rigid-body prim under", gripper_root_path)
        print("Prims under gripper root:")
        root = stage.GetPrimAtPath(Sdf.Path(gripper_root_path))
        for p in root.Traverse():
            print(" -", p.GetPath(), p.GetTypeName())
        raise RuntimeError("Failed to locate gripper rigid-body prim; update body1 path accordingly.")

    # create fixed joint between flange rigid body and gripper rigid body
    joint_path = Sdf.Path("/World/Origin2/Robot2_wrist3_gripper_fixed_joint")
    fixed = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(body1)])

    # return the scene information
    scene_entities = {
        "ur5e1": ur5e1,
        "ur5e2": ur5e2,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            )
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
