"""
Policy inference wrapper for TorchScript LSTM models – **V6** (24-dim obs, 12-dim actions).

Same observation / action space as V3, but the exported model is an LSTM
(ActorCriticRecurrent).  The JIT module stores hidden / cell state internally
and exposes a ``reset()`` method to clear memory between episodes.

Key differences from V3 (MLP):
  - ``reset()`` must be called before first inference and whenever the
    episode context should be cleared (e.g. new goal, homing, …)
  - The forward pass is stateful: hidden state is updated in-place inside
    the exported module.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union


class PolicyInferenceLSTM:
    """Wrapper for loading and running TorchScript LSTM policy inference (V6)."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        observation_dim: int = 24,
        action_dim: int = 12,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self._load_model()
        self._verify_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"[V6-LSTM] Loading policy from: {self.model_path}")

        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

        # Check that this is indeed a recurrent model
        has_reset = hasattr(self.model, "reset")
        print(f"[V6-LSTM] Policy loaded on device: {self.device}")
        print(f"[V6-LSTM] Recurrent model (has reset): {has_reset}")

        if not has_reset:
            print(
                "[V6-LSTM] WARNING: Model has no reset() method. "
                "This may be an MLP model exported without LSTM support."
            )

    def _verify_model(self):
        """Verify model dimensions with a dummy forward pass."""
        print("[V6-LSTM] Verifying model dimensions...")

        # Reset hidden state before verification
        self.reset()

        test_obs = torch.zeros(
            (1, self.observation_dim), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            try:
                output = self.model(test_obs)
                if isinstance(output, tuple):
                    actions = output[0]
                else:
                    actions = output

                actual_action_dim = actions.shape[-1]
                if actual_action_dim != self.action_dim:
                    print(
                        f"[V6-LSTM] WARNING: Expected action_dim={self.action_dim}, "
                        f"got {actual_action_dim}"
                    )
                    self.action_dim = actual_action_dim

                print(
                    f"[V6-LSTM] Model verified: obs_dim={self.observation_dim} "
                    f"-> action_dim={self.action_dim}"
                )
            except Exception as e:
                raise RuntimeError(f"[V6-LSTM] Model verification failed: {e}")

        # Reset again after verification to clear traces from test input
        self.reset()

    def reset(self):
        """Reset LSTM hidden/cell state to zeros.

        Must be called:
          - Before the first inference
          - When the episode context changes (new goal, homing, etc.)
        """
        if hasattr(self.model, "reset"):
            self.model.reset()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Run one step of LSTM inference and return the full 12-dim action vector.

        The internal hidden/cell state is updated in-place by the model.

        Returns:
            actions [12]: first 6 = position increments, last 6 = velocity targets
        """
        squeeze_output = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
            squeeze_output = True

        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)

        with torch.no_grad():
            output = self.model(obs_tensor)
            if isinstance(output, tuple):
                actions = output[0]
            else:
                actions = output

        actions_np = actions.cpu().numpy()
        if squeeze_output:
            actions_np = actions_np.squeeze(0)

        return actions_np


def load_policy(
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> PolicyInferenceLSTM:
    """Convenience function to load a V6 LSTM policy.

    Args:
        model_path: Path to model. If None, raises an error.
        device: Device to run on.

    Returns:
        PolicyInferenceLSTM instance (24-dim obs → 12-dim actions, stateful)
    """
    if model_path is None:
        raise RuntimeError("No model path provided. Please specify --model path/to/policy.pt")

    return PolicyInferenceLSTM(
        model_path=model_path,
        device=device,
        observation_dim=24,
        action_dim=12,
    )


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LSTM policy inference (v6)")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of sequential inference steps (to test LSTM state evolution)",
    )
    args = parser.parse_args()

    policy = load_policy(model_path=args.model, device=args.device)

    print(f"\nTesting {args.steps} sequential steps (LSTM state should evolve)...")
    policy.reset()

    for step in range(args.steps):
        obs = np.random.randn(24).astype(np.float32) * 0.1
        action = policy.get_action(obs)

        print(
            f"  Step {step:2d}: "
            f"pos_act=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},...) "
            f"vel_act=({action[6]:+.3f},{action[7]:+.3f},{action[8]:+.3f},...) "
            f"range=[{action.min():.3f}, {action.max():.3f}]"
        )

    print("\nResetting LSTM memory and re-running with same obs...")
    policy.reset()
    obs = np.zeros(24, dtype=np.float32)
    a1 = policy.get_action(obs)
    policy.reset()
    a2 = policy.get_action(obs)
    print(f"  Same output after reset: {np.allclose(a1, a2, atol=1e-6)}")
