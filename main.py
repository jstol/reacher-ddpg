#!/usr/bin/env python
# coding: utf-8
"""Main entry point – uses DDPG to solve the "Reacher" Unity ML-Agents environment"""

# Third party imports
import click

# Project imports
from ddpg import train_model, view_agent


# Define CLI
@click.group(help='train/visualize a DDPG agent within the Unity ml-agents "Reacher" environment')
def cli():
    pass


@cli.command(help='Train a DDPG model')
def train():
    train_model()


@cli.command(help='View a pre-trained agent, saved at POLICY_PATH')
@click.argument('policy-path', type=str)
def view(policy_path):
    view_agent(policy_path)


if __name__ == '__main__':
    cli()
