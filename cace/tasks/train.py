from typing import Optional, Dict, List, Type, Any
import logging
import torch
from torch import nn
from .loss import GetLoss
from ..tools import to_numpy, tensor_dict_to_device

"""
This file contains the training loop for the neural network model.
"""

__all__ = ['TrainingTask']

class TrainingTask(nn.Module):
    def __init__(self, 
                model: nn.Module,
                losses: List[GetLoss],
                device: torch.device = torch.device('cpu'),
                optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_args: Optional[Dict[str, Any]] = None,
                scheduler_cls: Optional[Type] = None,
                scheduler_args: Optional[Dict[str, Any]] = None,
                max_grad_norm: float = 10,
                ):
        """
        Args:
            model: the neural network model
            losses: list of losses an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
        """
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.losses = nn.ModuleList(losses)
        self.optimizer = optimizer_cls(self.parameters(), **optimizer_args)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_args) if scheduler_cls else None
        self.max_grad_norm = max_grad_norm

        self.grad_enabled = len(self.model.required_derivatives) > 0

    def forward(self, data, training: bool):
        return self.model(data, training=training)

    def loss_fn(self, pred, batch):
        loss = 0.0
        for eachloss in self.losses:
            loss += eachloss.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, subset, pred, batch):
        for eachloss in self.losses:
            eachloss.update_metrics(subset, pred, batch)

    def retrieve_metrics(self, subset):
        for eachloss in self.losses:
            for metric_name, metric in eachloss.metrics[subset].items():
                metric_now = to_numpy(torch.mean(torch.stack(metric))).item()
                print(
                    f'{subset}_{eachloss.name}_{metric_name}: {metric_now}',
                )
                logging.info(
                    f'{subset}_{eachloss.name}_{metric_name}: {metric_now}',
                )
            eachloss.clear_metric(subset)

    def train_step(self, batch, screen_nan: bool = True):
        torch.set_grad_enabled(True)

        batch.to(self.device)
        batch_dict = batch.to_dict()

        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(batch_dict, training=True)
        loss = self.loss_fn(pred, batch)
        loss.backward()

        # Print gradients for debugging purposes
        """
        for name, param in self.model.named_parameters():
            print(f"{name} requires grad: {param.requires_grad}")
            if param.requires_grad:
                print(f"Gradient of Loss w.r.t {name}: {param.grad}")
        """

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        normal = True
        if screen_nan:
            for param in self.model.parameters():
                if param.requires_grad and not torch.isfinite(param.grad).all():
                    normal = False
                    logging.info(f'!nan gradient!')
        if normal: 
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        #self.log_metrics('train', pred, batch)
        #return loss.item()
        return to_numpy(loss).item()

    def validate(self, val_loader):
        torch.set_grad_enabled(self.grad_enabled)
        
        self.eval()
        total_loss = 0
        for batch in val_loader:
            batch.to(self.device)
            batch_dict = batch.to_dict()
            pred = self.forward(batch_dict, training=False)
            # MACE put both on cpus, dunno why, trying it out
            batch = batch.cpu()
            pred = tensor_dict_to_device(pred, device=torch.device("cpu"))

            loss = to_numpy(self.loss_fn(pred, batch))
            total_loss += loss.item()
            self.log_metrics('val', pred, batch)
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs, val_stride: int = 1, screen_nan: bool = True):
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch, screen_nan=screen_nan)
                total_loss += loss
            avg_loss = total_loss / len(train_loader)
            #self.retrieve_metrics('train')
            if epoch % val_stride == 0:
                val_loss = self.validate(val_loader)
                self.retrieve_metrics('val')
                print(f'Epoch {epoch}, Train Loss: {avg_loss}, Val Loss: {val_loss}')
                logging.info(f'Epoch {epoch}, Train Loss: {avg_loss}, Val Loss: {val_loss}')

    def save_model(self, path: str):
        torch.save(self.model.to(torch.device("cpu")), path)
