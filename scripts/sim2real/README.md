# Sim2Real Policy Deployment for UR5e

Ce dossier contient les scripts pour déployer une politique entraînée en simulation (IsaacSim) sur le robot réel via ROS2.

## 📁 Structure des fichiers

```
scripts/sim2real/
├── observation_builder.py   # Construction et normalisation des observations
├── policy_inference.py      # Chargement et inférence TorchScript
├── sim2real_node.py        # Nœud ROS2 principal
├── goal_publisher.py       # Publieur de pose cible pour tests
├── launch_sim2real.bash    # Script de lancement
└── README.md               # Ce fichier
```

## 🔧 Prérequis

1. **ROS2 Humble** installé
2. **Workspace ROS2** configuré (`~/wwro_ws`)
3. **Politique entraînée** exportée en TorchScript (`.pt`)
4. **PyTorch** installé

## 🚀 Utilisation rapide

### 1. Lancer les contrôleurs du robot

```bash
# Dans un terminal, lancer le driver UR avec les contrôleurs
source ~/wwro_ws/install/local_setup.bash
ros2 launch ur_robot_driver ur_control.launch.py ...
```

### 2. Lancer le nœud sim2real

```bash
cd /home/robots/Woodworking_Simulation/scripts/sim2real
chmod +x launch_sim2real.bash
./launch_sim2real.bash --robot gripper
```

### 3. Publier une pose cible

```bash
# Dans un autre terminal
source ~/wwro_ws/install/local_setup.bash
cd /home/robots/Woodworking_Simulation/scripts/sim2real

# Mode statique
python3 goal_publisher.py --x 0.3 --y 0.0 --z 0.4

# Mode interactif
python3 goal_publisher.py --interactive

# Vue d'ensemble RViz de tous les goals d'un fichier
python3 goal_publisher.py --goals-file ../benchmark_settings/goals_handmade.json --goal-overview
```

En mode `--goal-overview`, le script ne publie pas de cible active sur `/goal_pose`.
Il publie uniquement un `MarkerArray` RViz persistant sur `/visualization_marker_array`, avec un repère et un label pour chaque entrée du fichier JSON.

### 4. Afficher les goals dans RViz

```bash
ros2 run rviz2 rviz2
```

Dans RViz:

1. Régler `Fixed Frame` sur `table`
2. Pour le mode standard, ajouter un affichage `Marker` sur `/visualization_marker`
3. Pour `--goal-overview`, ajouter un affichage `MarkerArray` sur `/visualization_marker_array`

Le mode standard affiche le goal actif.
Le mode `--goal-overview` affiche tous les goals du fichier simultanément, de manière persistante.

### 5. Afficher la trajectoire TCP complète (trace)

Si un topic de pose TCP est deja disponible (par ex.
`/gripper_tcp_pose_broadcaster/pose` en `geometry_msgs/PoseStamped`), vous
pouvez publier une trajectoire cumulative en `nav_msgs/Path` pour RViz :

```bash
source ~/wwro_ws/install/local_setup.bash
cd /home/robots/Woodworking_Simulation/scripts/sim2real

python3 ee_path_from_pose.py \
    --input-topic /gripper_tcp_pose_broadcaster/pose \
    --output-topic /ee_path \
    --max-points 5000 \
    --min-dt 0.03
```

Dans RViz:

1. Régler `Fixed Frame` sur `gripper_base` (ou une frame monde stable)
2. Ajouter un affichage `Path` sur `/ee_path`
3. Garder la trace visible pour comparer convergence stable vs oscillations

Option reset manuel de la trace:

```bash
ros2 service call /ee_path/clear std_srvs/srv/Empty {}
```

## 📊 Architecture des observations

Le vecteur d'observation (19 dimensions) est construit exactement comme dans IsaacSim :

| Index | Composant | Dimension | Normalisation |
|-------|-----------|-----------|---------------|
| 0-2 | `pos_error` | 3 | `(goal - ee) / 0.85` |
| 3-6 | `quat_error` | 4 | `goal_quat * ee_quat⁻¹` |
| 7-12 | `joint_pos` | 6 | `2*(pos-lower)/(upper-lower) - 1` |
| 13-18 | `joint_vel` | 6 | `vel / 3.14` |

### Constantes de normalisation

```python
MAX_REACH = 0.85      # Portée UR5e ~850mm
MAX_JOINT_VEL = 3.14  # ~180°/s

# Limites articulaires (elbow contraint par câble!)
JOINT_LIMITS = {
    "shoulder_pan":  (-2π, 2π),
    "shoulder_lift": (-2π, 2π),
    "elbow":         (-π, π),    # ⚠️ Contrainte câble!
    "wrist_1":       (-2π, 2π),
    "wrist_2":       (-2π, 2π),
    "wrist_3":       (-2π, 2π),
}
```

## ⚙️ Topics ROS2 utilisés

### Souscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Positions et vitesses articulaires |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Pose cible de l'effecteur |

### TF Frames

| Frame | Description |
|-------|-------------|
| `table` | Base de l'environnement |
| `gripper_wrist_3_link` | Effecteur terminal |

### Actions

| Action | Type | Description |
|--------|------|-------------|
| `/gripper_scaled_joint_trajectory_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | Commande trajectoire |

## 🎯 Paramètres de la politique

Les actions sont des deltas de position articulaire, scalés comme suit :

```python
ACTION_SCALE = 7.5
DOF_VELOCITY_SCALE = 0.1
dt = 1/60  # 60Hz

# Calcul des nouvelles cibles
inc = dt * DOF_VELOCITY_SCALE * ACTION_SCALE * actions
targets = clamp(targets + inc, lower, upper)
```

## 🔍 Déboggage

### Vérifier les topics

```bash
source ~/wwro_ws/install/local_setup.bash

# Liste des topics
ros2 topic list

# Écouter les joint states
ros2 topic echo /joint_states

# Vérifier les TF
ros2 run tf2_tools view_frames
```

### Tester la politique en isolation

```bash
cd /home/robots/Woodworking_Simulation/scripts/sim2real
python3 policy_inference.py --model ../../pretrained_models/exported/policy.pt
```

## ⚠️ Sécurité

1. **Toujours** avoir le bouton d'arrêt d'urgence à portée
2. Commencer avec des gains faibles et augmenter progressivement
3. Vérifier les limites articulaires avant déploiement
4. Tester d'abord en mode simulation ou avec le robot en position sûre

## 📝 Notes

- La politique attend des observations normalisées exactement comme en simulation
- La frame de référence pour la pose cible est `gripper_base_link`
- Le contrôleur `scaled_joint_trajectory_controller` respecte les limites de vitesse du robot
