# Benchmark pose/orientation

## Spécifications

- Comparer une policy en simulation et sur le setup réel.
- Utiliser une policy pose / orientation.
- Démarrer le benchmark depuis `home`.
- En simulation, supprimer la logique de spawn aléatoire pour le benchmark.
- Utiliser 10 goals définis manuellement.
- Format des goals : `[x, y, z, qw, qx, qy, qz]`.
- Campagne déterministe avec une seed.
- Temps maximal par goal : 5 s pour le benchmark sim actuel.
- `in_area` : 5 cm et 15°.
- `on_goal` : 2 cm et 10°.

## Métriques demandées

- Temps jusqu'à `in_area`.
- Temps jusqu'à `on_goal`.
- Moyenne de la distance TCP-goal pendant `in_area`.
- Moyenne de l'angle TCP-goal pendant `in_area`.

## Avancement

### Fait

- Ajout de [scripts/rsl_rl/test.py](scripts/rsl_rl/test.py) comme runner sim dérivé de `play.py`.
- Runner sim simplifié pour n'exposer que les arguments utiles au benchmark.
- Désactivation de la domain randomization en dur dans le runner sim.
- Ajout d'un reset home déterministe via `reset_range=0.0` en dur dans le runner sim.
- Simplification du reset V4 vers un reset home unique.
- Ajout d'un mode booléen `deterministic_goal_sampling` dans [source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v4/pose_orientation_sim2real_v4.py](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v4/pose_orientation_sim2real_v4.py).
- Ajout d'un tenseur interne minimal pour stocker des goals explicites.
- Ajout d'un fichier par défaut [scripts/benchmark_settings/default_benchmark_goals.json](scripts/benchmark_settings/default_benchmark_goals.json).
- Ajout d'une CLI `--goals-file` pour injecter les goals explicites dans le runner sim.
- Implémentation des métriques `in_area` et `on_goal` dans [scripts/rsl_rl/test.py](scripts/rsl_rl/test.py).
- Export automatique d'un JSON de benchmark dans `logs/benchmarks/`.
- Le runner sim utilise maintenant la longueur réelle du tuple de goals pour déterminer le nombre de goals à exécuter.
- Le runner sim exécute un seul passage complet sur la liste de goals, sans reset intermédiaire de l'environnement.
- Le runner sim accepte maintenant soit un checkpoint `.pt`, soit un fichier JSON de checkpoints comme [scripts/benchmark_settings/default_benchmark_checkpoints.json](scripts/benchmark_settings/default_benchmark_checkpoints.json).

### En cours / à faire

- Ajouter le runner benchmark pour le setup réel.
- Ajouter un vrai fichier de config benchmark dédié.

## Exemple CLI actuel

```bash
python scripts/rsl_rl/test.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v4 \
  --checkpoint logs/rsl_rl/.../model_1499.pt \
  --num_envs 1 \
  --seed 12 \
  --goals-file scripts/benchmark_settings/default_benchmark_goals.json
```

## Format JSON pour plusieurs checkpoints

```json
{
  "checkpoints": [
    "logs/rsl_rl/run_a/model_1499.pt",
    "logs/rsl_rl/run_b/model_1499.pt"
  ]
}
```
