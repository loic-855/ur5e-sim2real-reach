"""
Policy inference wrapper for TorchScript models – **V2** (24-dim observations).
Loads the exported policy and provides a simple interface for inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union


class PolicyInference:
    """Wrapper for loading and running TorchScript policy inference."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        observation_dim: int = 24,
        action_dim: int = 6,
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

        print(f"Loading policy from: {self.model_path}")

        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

        print(f"Policy loaded successfully on device: {self.device}")

    def _verify_model(self):
        print("Verifying model dimensions...")
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
                        f"WARNING: Expected action_dim={self.action_dim}, got {actual_action_dim}"
                    )
                    self.action_dim = actual_action_dim

                print(
                    f"Model verified: obs_dim={self.observation_dim} -> action_dim={self.action_dim}"
                )
            except Exception as e:
                raise RuntimeError(f"Model verification failed: {e}")

    def get_action(self, observation: np.ndarray) -> np.ndarray:
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
) -> PolicyInference:
    """Convenience function to load policy.

    Args:
        model_path: Path to model. If None, uses a default exported policy.
        device: Device to run on.

    Returns:
        PolicyInference instance
    """
    if model_path is None:
        raise RuntimeError("No default model path defined. Please provide a model_path.")

    return PolicyInference(
        model_path=model_path,
        device=device,
        observation_dim=24,
        action_dim=6,
    )


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test policy inference (v2)")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    args = parser.parse_args()

    policy = load_policy(model_path=args.model, device=args.device)

    print("\nTesting with random observation...")
    obs = np.random.randn(24).astype(np.float32) * 0.1
    action = policy.get_action(obs)

    print(f"Observation shape: {obs.shape}")
    print(f"Action: {action}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
