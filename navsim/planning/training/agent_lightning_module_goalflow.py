from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor
import os
import numpy as np

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k,v in loss.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=len(batch[0]))
        return loss['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")
    
    def test_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        features, targets = batch
        prediction = self.agent.forward(features,targets)
        if self.agent._config.generate=='goal_score':
            log_dir = self.logger.log_dir if self.logger else "./default_log_dir"
            navis_dir_path = os.path.join(log_dir, "navis")
            os.makedirs(navis_dir_path, exist_ok=True)
            im_path=os.path.join(navis_dir_path,'im')
            dac_path=os.path.join(navis_dir_path,'dac')
            os.makedirs(im_path, exist_ok=True)
            os.makedirs(dac_path,exist_ok=True)
            batch_size=prediction['im_scores'].shape[0]
            for i in range(batch_size):
                token=features['token'][i]
                im_score=prediction['im_scores'][i].squeeze().cpu().numpy()
                dac_score=prediction['dac_scores'][i].squeeze().cpu().numpy()
                np.save(f'{dac_path}/{token}.npy',dac_score)
                np.save(f'{im_path}/{token}.npy',im_score)

        elif self.agent._config.generate=='trajectory':
            log_dir = self.logger.log_dir if self.logger else "./default_log_dir"
            trajs_dir_path = os.path.join(log_dir, "trajs")
            os.makedirs(trajs_dir_path, exist_ok=True)
            
            trajs_num = prediction['trajectory'].shape[0]
            for i in range(trajs_num):
                token = features['token'][i]
                traj = prediction['trajectory'][i].squeeze(0).cpu().numpy()
                np.save(f'{trajs_dir_path}/{token}.npy', traj)
        else:
            raise Exception('generate should be in (goal_score,trajectory)')
        return prediction

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
