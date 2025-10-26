import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AlohaStateExtractor(BaseFeaturesExtractor):
    """
    Custom MLP feature extractor for ALOHA state observations.

    Processes agent positions and object states (poses) and extracts features for the policy.
    The state observation contains:
    - agent_pos: joint positions of both arms (14 dims)
    - env_state: object poses (7 dims for cube, 14 dims for peg+socket)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Get dimensions from observation space
        agent_pos_dim = observation_space.spaces["agent_pos"].shape[0]
        agent_vel_dim = observation_space.spaces["agent_vel"].shape[0]
        env_state_dim = observation_space.spaces["env_state"].shape[0]
        total_dim = agent_pos_dim + agent_vel_dim + env_state_dim

        # MLP architecture for state processing
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations is a dict with 'agent_pos' and 'env_state' keys
        agent_pos = observations["agent_pos"]
        agent_vel = observations["agent_vel"]
        env_state = observations["env_state"]

        # Concatenate all state information
        state = torch.cat([agent_pos, agent_vel, env_state], dim=1)

        # Extract features through MLP
        features = self.mlp(state)

        return features


class AlohaImageExtractor(BaseFeaturesExtractor):
    """
    CURRENTLY NOT USED

    Custom CNN feature extractor for ALOHA image observations.

    Processes the 'top' camera image and extracts features for the policy.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # Calculate the output dimension
        super().__init__(observation_space, features_dim)

        # Get image shape from observation space
        image_space = observation_space.spaces["top"]

        # Handle both CHW and HWC formats
        # Check if format is CHW (channels first) or HWC (channels last)
        if image_space.shape[0] == 3 or image_space.shape[0] == 1:
            # CHW format: (channels, height, width)
            n_input_channels = image_space.shape[0]
            self.is_channels_first = True
        else:
            # HWC format: (height, width, channels)
            n_input_channels = image_space.shape[2]
            self.is_channels_first = False

        # CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create dummy observation
            sample_image = torch.as_tensor(image_space.sample()[None]).float()

            # Convert to NCHW if needed
            if not self.is_channels_first:
                sample_image = sample_image.permute(0, 3, 1, 2)  # NHWC -> NCHW

            n_flatten = self.cnn(sample_image).shape[1]

        # Final linear layer to get desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations is a dict with 'top' key containing images
        images = observations["top"]

        # Normalize pixel values to [0, 1]
        images = images.float() / 255.0

        # Convert from NHWC to NCHW format if needed
        if not self.is_channels_first:
            images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Extract features
        features = self.cnn(images)
        features = self.linear(features)

        return features
