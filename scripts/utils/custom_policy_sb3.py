import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th

class CNN_GAP_BN(BaseFeaturesExtractor):
    """
    Feature extractor combining CNN and state vector for image input.
    :param observation_space: (gym.Space) Observation space containing image and state info.
    :param features_dim: (int) Output dimension of the feature extractor.
    :param state_feature_dim: (int) Number of elements in the state vector.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 33, state_feature_dim=8):
        super(CNN_GAP_BN, self).__init__(observation_space, features_dim)
        assert state_feature_dim > 0

        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = 25
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]
        x = self.conv1(depth_img)
        x = self.conv2(x)
        x = self.conv3(x)
        cnn_feature = self.gap_layer(x).squeeze(-1).squeeze(-1)  # shape: [B, feature_num_cnn]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        state_feature = ((state_feature / 255.0) * 2) - 1  # normalize to [-1, 1]

        combined_feature = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = combined_feature
        return combined_feature


class Noise_Actor(CNN_GAP_BN):
    """
    CNN-based feature extractor for Actor in asymmetric training. Uses noisy observation inputs.
    """
    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 1:2, :, :]
        x = self.conv1(depth_img)
        x = self.conv2(x)
        x = self.conv3(x)
        cnn_feature = self.gap_layer(x).squeeze(-1).squeeze(-1)

        state_feature = observations[:, 2, 0, self.feature_num_state:2*self.feature_num_state]
        state_feature = ((state_feature / 255.0) * 2) - 1

        combined_feature = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = combined_feature
        return combined_feature


class Noise_Critic(Noise_Actor):
    """
    CNN-based feature extractor for Critic. Uses noisy depth image and partial state info.
    """
    pass


class Noise_Critic_Symmetry(Noise_Critic):
    """
    Symmetric Critic for privileged learning. Uses ground-truth depth image and state.
    """
    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]
        x = self.conv1(depth_img)
        x = self.conv2(x)
        x = self.conv3(x)
        cnn_feature = self.gap_layer(x).squeeze(-1).squeeze(-1)

        state_feature = observations[:, 2, 0, 0:self.feature_num_state]
        state_feature = ((state_feature / 255.0) * 2) - 1

        combined_feature = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = combined_feature
        return combined_feature


class Noise_Critic_Asymmetry(BaseFeaturesExtractor):
    """
    Asymmetric Critic that processes two depth images separately for privileged learning.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(Noise_Critic_Asymmetry, self).__init__(observation_space, features_dim)
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim * 2

        # First CNN branch
        self.conv1a = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3a = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn // 2, 3, 1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.gap_layer_a = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        # Second CNN branch
        self.conv1b = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn // 2, 3, 1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.gap_layer_b = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.feature_all = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # First depth image path
        x1 = self.conv1a(observations[:, 0:1, :, :])
        x1 = self.conv2a(x1)
        x1 = self.conv3a(x1)
        cnn_feature_a = self.gap_layer_a(x1).squeeze(-1).squeeze(-1)

        # Second depth image path
        x2 = self.conv1b(observations[:, 1:2, :, :])
        x2 = self.conv2b(x2)
        x2 = self.conv3b(x2)
        cnn_feature_b = self.gap_layer_b(x2).squeeze(-1).squeeze(-1)

        # Combine CNN features
        cnn_features = th.cat((cnn_feature_a, cnn_feature_b), dim=1)

        # Normalize state features
        state_feature = observations[:, 2, 0, 0:2*self.feature_num_state]
        state_feature = ((state_feature / 255.0) * 2) - 1

        combined_feature = th.cat((cnn_features, state_feature), dim=1)
        self.feature_all = combined_feature
        return combined_feature
