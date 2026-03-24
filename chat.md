## 🔬 Analyse critique de la configuration V1

### 1. Gains PD de l'actuateur — **Cause principale des oscillations**

Les gains du `DelayedPDActuator` sont **sous-amortis** pour un contrôle en position :

| Joint group | Stiffness ($K_p$) | Damping ($K_d$) | Ratio $\zeta = \frac{K_d}{2\sqrt{K_p}}$ |
|---|---|---|---|
| Shoulder (pan/lift/elbow) | 160 | 28 | ~1.11 ✅ |
| Wrist 1 | 125 | 19 | ~0.85 ⚠️ |
| Wrist 2 | 100 | 16 | ~0.80 ⚠️ |
| Wrist 3 | 80 | 14 | ~0.78 ⚠️ |

Les wrists sont **sous-critiquement amortis** ($\zeta < 1$), ce qui explique directement les oscillations observées — surtout wrist_3 qui contrôle l'orientation finale. Pour un contrôle en position stable, il faut $\zeta \geq 1$ (amortissement critique ou sur-amorti).

**Recommandation** : Augmenter le damping des wrists. Par exemple :
- Wrist 1 : $K_d = 2\sqrt{125} \approx 22.4$ → utiliser **24**
- Wrist 2 : $K_d = 2\sqrt{100} = 20$ → utiliser **22**
- Wrist 3 : $K_d = 2\sqrt{80} \approx 17.9$ → utiliser **20**

Ou mieux : augmenter les stiffness aussi pour un tracking plus rapide (ex: 200/150/120/100 avec damping proportionnel).
-> essai en changeant l'amortissement et augmente le max torque des gros joints
---

### 2. Méthode de contrôle — **Delta position + action_scale trop élevé**

La boucle de contrôle dans `_pre_physics_step` :
```
delta = dt * action_scale * actions   →   0.01 * 4.0 * [-1,1] = ±0.04 rad/step
```

Cela signifie que la policy peut commander **±0.04 rad par step** (soit ~2.3°/step à 100 Hz). Combiné avec les gains PD sous-amortis, cela crée un cycle :
1. La policy commande un grand delta
2. L'actuateur PD essaie de suivre mais oscille autour du target
3. La policy reçoit des observations bruitées par l'oscillation
4. Elle surcompense → oscillation amplifiée

**Recommandations** :
- **Réduire `action_scale` à 1.5–2.0** pour limiter les deltas par step
- **Ajouter les actions précédentes dans l'observation** (`self.actions` ou `self.q_des`) — actuellement la policy est "aveugle" à ce qu'elle a commandé au step précédent, ce qui empêche un comportement lisse
- **Ajouter un smoothing exponentiel** sur `q_des` : `q_des = α * q_des_new + (1-α) * q_des_old` avec $\alpha \approx 0.3$
-> action scale 1.5 + smoothing (1st order filter)
---

### 3. Espace d'observation — **Incomplet pour un contrôle lisse**

Actuellement 19 dims : `to_target(3) + ori_error(4) + joint_pos(6) + joint_vel(6)`

**Observations manquantes critiques** :
| Observation | Dims | Pourquoi |
|---|---|---|
| Actions précédentes | 6 | Permet à la policy d'apprendre la continuité temporelle |
| `q_des` actuel (target courant) | 6 | La policy ne sait pas où elle a commandé le robot |
| Distance au but (scalaire) | 1 | Signal clair de proximité |
| Temps restant dans l'épisode | 1 | Permet d'adapter la stratégie (aller vite puis stabiliser) |

Ajouter au minimum **les actions précédentes** (augmenter `observation_space` à 25). C'est quasi-standard pour tout contrôle de robot RL avec actuateurs réalistes.

-> ajouté juste les previous actions, on verra plus tard pour plus
---

### 4. Structure de récompense — **Signaux contradictoires et shaping sous-optimal**

La récompense actuelle :
$$r = 0.5 \cdot (1 - \tanh(\frac{d}{\sigma})) - 0.2 \cdot d - 0.13 \cdot e_{ori} - 0.01 \cdot \|a\|^2 - 0.5 \cdot e_{elbow}$$

**Problèmes** :
- **`std = 0.1`** pour le tanh kernel est **très serré** : à 10 cm du goal, $\tanh(0.1/0.1) = 0.76$, donc la reward position est quasi-nulle dès 20cm. La policy reçoit peu de gradient de shaping quand elle est loin. → **Augmenter `std` à 0.2–0.3** pour un gradient plus progressif.
- **Double pénalisation de la position** : Le terme tanh et le terme linéaire $-0.2 \cdot d$ pénalisent tous deux la distance. Le tanh sature vite, donc loin du goal la policy ne voit que le signal linéaire (faible). → Choisir l'un ou l'autre, ou utiliser un tanh avec `std` plus large comme seul signal.
- **Pénalité d'orientation trop faible** ($-0.13$) par rapport à la position ($+0.5 - 0.2$). Puisque l'erreur d'orientation (en radians) peut être jusqu'à $\pi$, le gradient d'orientation est noyé. → **Augmenter à $-0.3$ ou $-0.5$** et introduire un bonus d'orientation séparé (tanh kernel aussi).
- **Pas de reward pour la stabilité** : Aucun bonus pour rester proche du goal de manière stable. → Ajouter un **bonus de "dwell"** : si $d < 0.05$ et $e_{ori} < 0.1$, donner un bonus.
- **Action penalty trop faible** ($-0.01$) pour décourager les oscillations. → **Augmenter à $-0.05$** et/ou ajouter une **pénalité sur le jerk** (différence d'actions consécutives) : $-\lambda \|a_t - a_{t-1}\|^2$.

**Structure suggérée** :
$$r = w_1 \cdot (1 - \tanh(\frac{d}{0.25})) + w_2 \cdot (1 - \tanh(\frac{e_{ori}}{0.5})) - w_3 \|a\|^2 - w_4 \|a_t - a_{t-1}\|^2 - w_5 \cdot e_{elbow} + w_6 \cdot \mathbb{1}_{close}$$

-> reward un peu comme le robot franka, pas comme proposé
---


### 5. Hyperparamètres PPO — **Exploration insuffisante**

| Paramètre | Valeur actuelle | Problème | Suggestion |
|---|---|---|---|
| `init_noise_std` | 1.0 | Très élevé au début, cause des actions erratiques | **0.5** |
| `entropy_coef` | 0.0 | **Aucune exploration** au-delà du bruit gaussien | **0.005–0.01** |
| `num_steps_per_env` | 16 | Court horizon de rollout pour un épisode de 400 steps (4s @ 100Hz) | **24–32** |
| `desired_kl` | 0.008 | Très conservateur, ralentit l'apprentissage | **0.01–0.016** |
| `learning_rate` | 5e-4 | OK mais couplé avec le KL adaptatif, peut être réduit trop vite | Garder, mais monitorer le LR effectif |
| `actor_hidden_dims` | [128, 64] | Petit réseau, OK pour 19 obs mais sera limite avec plus d'obs | **[256, 128, 64]** si ajout d'observations |
| `max_iterations` | 600 | **Trop peu** pour un problème 6-DOF avec orientation | **2000–5000** |

---

### 6. Domain Randomization — **Totalement absent**

C'est le point le plus critique pour le sim-to-real. **Aucune randomisation** n'est actuellement appliquée. Voici ce qu'il faut ajouter progressivement :

**Priorité haute (indispensable pour sim2real)** :
| Paramètre | Plage suggérée | Méthode |
|---|---|---|
| Gains PD ($K_p$, $K_d$) | ±30% de la valeur nominale | Randomiser à chaque reset |
| Délai de l'actuateur | 1–4 steps (10–40ms) | `min_delay=1, max_delay=4` |
| Friction des joints | ±50% | Ajouter `friction` dans `DelayedPDActuatorCfg` |
| Masse des liens | ±15% | `mass_props` randomization |
| Offset de position TCP | ±1cm | Randomiser l'offset du FrameTransformer |

**Priorité moyenne** :
| Paramètre | Plage | Détail |
|---|---|---|
| Bruit d'observation | σ = 0.005 rad sur joint_pos, 0.01 rad/s sur joint_vel | Gaussian additif |
| Latence de la policy | 0–2 steps supplémentaires | Retarder l'application des actions |
| Gravité | ±0.5 m/s² | Environnements inclinés |
| Bruit sur les actions | ±0.02 rad | Gaussian additif sur `q_des` |

**Stratégie d'implémentation** : Utiliser le system `events` d'Isaac Lab avec `EventTermCfg` pour randomiser à chaque reset. Commencer avec les gains PD + friction, puis ajouter le delay progressivement.

---

### 7. Reset et Goal Sampling — **Quelques incohérences**

- **`reset_frac = 0.8`** : Seulement 80% des envs sont vraiment reset en joint position. Les 20% restants gardent leur position finale → risque de "lazy policies" qui apprennent à ne pas bouger dans les cas faciles.  → **Mettre à 1.0** pendant le développement.
- **Bruit de reset : `±0.4 rad`** autour de la position par défaut : c'est correct mais pourrait être augmenté à **±0.8** pour plus de diversité.
- **Goal sampling** : Le radius [0.3, 0.75] et la hauteur [0.1, 0.6] sont raisonnables, mais certains goals pourraient être hors d'atteinte en orientation (combinaison rayon + tilt). → Ajouter un **filtre de faisabilité** via IK ou au minimum vérifier que le goal est dans l'enveloppe de travail.

---

### 8. Simulation — **dt et decimation**

- `sim.dt = 0.005` (200 Hz) avec `decimation = 2` → 100 Hz policy
- C'est **trop lent** pour un `DelayedPDActuator` avec delay. Le PD controller n'a que 2 substeps par action pour stabiliser. 
- **Recommandation** : Passer à `dt = 1/500` (500 Hz) avec `decimation = 5` → policy reste à 100 Hz mais l'actuateur PD a 5 substeps pour converger, bien plus stable.

---

### 📋 Plan d'action priorisé

1. **🔴 Immédiat** : Corriger le damping des wrists pour $\zeta \geq 1$, réduire `action_scale` à 2.0
2. **🔴 Immédiat** : Ajouter les actions précédentes en observation + pénalité de jerk
3. **🟠 Court terme** : Augmenter `std` à 0.25, réduire `init_noise_std` à 0.5, ajouter `entropy_coef = 0.005`
4. **🟠 Court terme** : Passer à `dt=1/500, decimation=5` pour stabiliser le PD
5. **🟡 Moyen terme** : Restructurer la reward (tanh unique + bonus dwell)
6. **🟡 Moyen terme** : Domain randomization des gains PD ± 30%
7. **🟢 Long terme** : Introduire le delay progressivement (`max_delay=1` puis `2`, etc.)
8. **🟢 Long terme** : Randomisation complète (masse, friction, bruit, latence)

Veux-tu que j'implémente certaines de ces modifications ?