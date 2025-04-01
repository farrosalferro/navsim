from typing import Any, List, Dict, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.agents.goalflow.goalflow_model_navi import GoalFlowNaviModel
from navsim.agents.goalflow.goalflow_loss import goalflow_loss
from navsim.agents.goalflow.goalflow_features import (
    GoalFlowFeatureBuilder,
    GoalFlowTargetBuilder,
)
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig, Scene
from torch.optim.lr_scheduler import StepLR
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class GoalFlowNaviAgent(AbstractAgent):
    def __init__(
        self,
        config: GoalFlowConfig,
        lr: float,
        checkpoint_path: str = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._goalflow_model = GoalFlowNaviModel(config)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [GoalFlowTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [GoalFlowFeatureBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._goalflow_model(features,targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict:
        return goalflow_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._goalflow_model.parameters()),
            lr=self._lr
        )
        
        scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1)]
