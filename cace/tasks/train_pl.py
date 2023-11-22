from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn

from .loss import GetLoss
from ..models import AtomisticModel

__all__ = ["TrainingTask_PL"]

class TrainingTask_PL(pl.LightningModule):
    """
    The basic learning task using PL

    """

    def __init__(
        self,
        model: AtomisticModel,
        losses: List[GetLoss],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            losses: list of losses an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.losses = nn.ModuleList(losses)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        #self.save_hyperparameters()

    def forward(self, data: Dict[str, torch.Tensor], training=True):
        data = self.model(data, training=training)
        return data

    def loss_fn(self, pred, batch):
        loss = 0.0
        for eachloss in self.losses:
            loss += eachloss.calculate_loss(pred, batch)
        return loss

    def training_step(self, batch, batch_idx):

        pred = self.model(batch)
        loss = self.loss_fn(pred, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        if isinstance(batch, tuple) and len(batch) == 1 and isinstance(batch[0], dict):
            batch = batch[0]

        pred = self.model(batch)
        loss = self.loss_fn(pred, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        pred = self.model(batch)
        loss = self.loss_fn(pred, batch)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=0,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer_closure()
        #optimizer.step(closure=optimizer_closure)
        optimizer.step()

    def save_model(self, path: str):
        torch.save(self.model, path)
