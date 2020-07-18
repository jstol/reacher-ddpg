#!/usr/bin/env python
# coding: utf-8
"""Polcy/Q-Net (Actor/Critic) model definitions.

Note: A good overview of the DDPG model can be found here: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
"""

# Standard imports
from typing import (
    List,
    Optional,
)

# Third party imports
import torch
from torch import (
    nn,
    Tensor,
)


# Define Model Classes
class _FeedForward(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], final_activation: nn.Module = None):
        """Creates a generic feed forward network (with a real-valued output).

        Args:
            input_size: Size of input vector.
            output_size: Size of the (continuous) action space.
            hidden_sizes: List detailing the number/sizes of the hidden layers to use.
            final_activation: An activation that, if provided, gets added to the last layer.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        # Set up DNN with ReLU activations
        # Input
        layers = [
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_sizes[0]),
        ]

        # Hidden layers
        for h1, h2 in [(self.hidden_sizes[i], self.hidden_sizes[i + 1]) for i in range(len(self.hidden_sizes[:-1]))]:
            layers += [
                nn.Linear(h1, h2),
                nn.ReLU(),
            ]

        # Output
        layers += [
            nn.Linear(self.hidden_sizes[-1], self.output_size),
        ]

        if final_activation:
            layers.append(final_activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, input_: Tensor) -> Tensor:
        return self.layers(input_)

    def set_requires_grad(self, enabled: bool):
        for param in self.parameters():
            param.requires_grad = enabled


class QNet(_FeedForward):
    def __init__(self, state_size: int = 33, action_size: int = 4, hidden_sizes: Optional[List[int]] = None):
        """Creates a DQN to estimate action-values.

        Args:
            state_size: Size of the state space.
            action_size: Size of the (continuous) action space.
            hidden_sizes: List detailing the number/sizes of the hidden layers to use.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [32, 16, 8]

        input_size = self.state_size + self.action_size

        super().__init__(input_size, 1, self.hidden_sizes)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return super().forward(torch.cat((state, action), -1))


class PolicyNet(_FeedForward):
    def __init__(self, state_size: int = 33, action_size: int = 4, hidden_sizes: Optional[List[int]] = None):
        """Creates a Deep-Q Network to estimate action-values.

        Args:
            state_size: Size of the state space.
            hidden_sizes: List detailing the number/sizes of the hidden layers to use.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [32, 16, 8]

        super().__init__(self.state_size, self.action_size, self.hidden_sizes, final_activation=nn.Tanh())
