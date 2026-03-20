"""English
sim_gain_tuner_logger.py
------------------------

Simple Isaac Sim script to record commanded and observed joint positions
and velocities while the timeline is playing. Open this file in the Script
Editor (Python extension), press Play to record and Stop to export a
timestamped CSV into `SAVE_DIR`. Make sure `ROBOT_PRIM_PATH` points to the
robot prim in the Stage.
"""

import csv
import os
import omni.timeline
import omni.physx
from datetime import datetime
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import is_prim_path_valid

# --- CONFIGURATION ---
ROBOT_PRIM_PATH = "/World/ur5e_gripper_tcp_small"
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Woodworking_Simulation", "logs", "sim_gain_tuner")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- INITIALISATION ---
if not is_prim_path_valid(ROBOT_PRIM_PATH):
    print(f"ERREUR : Le chemin '{ROBOT_PRIM_PATH}' n'existe pas.")
else:
    # Utilisation d'une View (plus leger et evite les erreurs de duplication)
    robot_view = ArticulationView(prim_paths_expr=ROBOT_PRIM_PATH)
    robot_view.initialize()
    
    # Recuperation des noms des joints
    joint_names = robot_view.dof_names
    num_dof = robot_view.num_dof

    header = ['time']
    for name in joint_names:
        header.extend([f"{name}_pos_cmd", f"{name}_pos_obs", f"{name}_vel_cmd", f"{name}_vel_obs"])

    data_log = []

    def logging_callback(step_size):
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return
            
        t = timeline.get_current_time()
        
        # Lecture des etats (Observed)
        obs_pos = robot_view.get_joint_positions()[0]
        obs_vel = robot_view.get_joint_velocities()[0]
        
        # Lecture des commandes (Command)
        # On passe par l'interface physique pour eviter les erreurs de NoneType
        action = robot_view.get_applied_actions()
        cmd_pos = action.joint_positions[0] if action.joint_positions is not None else [0.0] * num_dof
        cmd_vel = action.joint_velocities[0] if action.joint_velocities is not None else [0.0] * num_dof

        row = [t]
        for i in range(num_dof):
            row.extend([cmd_pos[i], obs_pos[i], cmd_vel[i], obs_vel[i]])
        data_log.append(row)

    def export_to_csv():
        if not data_log:
            print("INFO : Pas de donnees a exporter.")
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_path = os.path.join(SAVE_DIR, f"{timestamp}_gain_tuner_plot.csv")
        
        with open(full_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_log)
        
        print(f"Donnees sauvegardees : {full_path}")
        data_log.clear()

    # --- GESTION DES CALLBACKS (SANS WORLD) ---
    # On utilise l'interface Physx directe pour le callback
    physx_subs = omni.physx.get_physx_interface().subscribe_physics_step_events(logging_callback)

    def on_timeline_event(event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            export_to_csv()

    timeline_sub = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(on_timeline_event)

    print(f"Pret : Robot detecte ({num_dof} joints). Lancement de l'enregistrement.")
