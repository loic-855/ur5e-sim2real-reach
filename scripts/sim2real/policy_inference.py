"""
Policy inference wrapper for TorchScript models.
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
        observation_dim: int = 26,
        action_dim: int = 6,
    ):
        """Initialize policy inference.
        
        Args:
            model_path: Path to TorchScript .pt file
            device: Device to run inference on ("cpu" or "cuda")
            observation_dim: Expected observation dimension
            action_dim: Expected action dimension
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Load model
        self._load_model()
        
        # Verify model dimensions
        self._verify_model()
        
    def _load_model(self):
        """Load TorchScript model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading policy from: {self.model_path}")
        
        # Load TorchScript model
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        
        print(f"Policy loaded successfully on device: {self.device}")
        
    def _verify_model(self):
        """Verify model input/output dimensions with a test inference."""
        print("Verifying model dimensions...")
        
        # Create test input
        test_obs = torch.zeros((1, self.observation_dim), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            try:
                output = self.model(test_obs)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    actions = output[0]  # Some policies return (actions, ...)
                else:
                    actions = output
                
                actual_action_dim = actions.shape[-1]
                
                if actual_action_dim != self.action_dim:
                    print(f"WARNING: Expected action_dim={self.action_dim}, got {actual_action_dim}")
                    self.action_dim = actual_action_dim
                
                print(f"Model verified: obs_dim={self.observation_dim} -> action_dim={self.action_dim}")
                
            except Exception as e:
                raise RuntimeError(f"Model verification failed: {e}")
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Run inference to get actions from observation.
        
        Args:
            observation: Observation vector [obs_dim] or [batch, obs_dim]
            
        Returns:
            Action vector [action_dim] or [batch, action_dim]
        """
        # Handle single observation
        squeeze_output = False
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
            squeeze_output = True
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(obs_tensor)
            
            # Handle different output formats
            if isinstance(output, tuple):
                actions = output[0]
            else:
                actions = output
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()
        
        # Squeeze if input was single observation
        if squeeze_output:
            actions_np = actions_np.squeeze(0)
        
        return actions_np


def load_policy(
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> PolicyInference:
    """Convenience function to load policy.
    
    Args:
        model_path: Path to model. If None, uses default exported policy.
        device: Device to run on ("cpu" or "cuda")
        
    Returns:
        PolicyInference instance
    """
    if model_path is None:
        # Default path relative to this file
        repo_root = Path(__file__).resolve().parents[2]
        #model_path = str(repo_root / "logs" / "rsl_rl" / "pose_orientation_sim2real_ext_nn" / "2026-02-13_10-43-10" / "exported" / "policy.pt")
        model_path = str(repo_root/"logs/rsl_rl/2026-02-16_00-59-56_actuators-high_domain_rand-current_network-ext3_action_rate-current/exported/policy.pt")
        #model_path = str(repo_root/"logs/rsl_rl/pose_orientation_sim2real_ext_nn_125hz/2026-02-13_14-35-41/exported/policy.pt")
    return PolicyInference(
        model_path=model_path,
        device=device,
        observation_dim=26,
        action_dim=6,
    )


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test policy inference")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    
    # Load policy
    policy = load_policy(model_path=args.model, device=args.device)
    
    # Test with random observation
    print("\nTesting with random observation...")
    obs = np.random.randn(26).astype(np.float32) * 0.1
    action = policy.get_action(obs)
    
    print(f"Observation: {obs}")
    print(f"Action: {action}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
