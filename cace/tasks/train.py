from typing import Optional, Dict, List, Type, Any

import torch
from torch import nn
from .loss import GetLoss
from ..models import AtomisticModel

"""
This file contains the training loop for the neural network model.
"""

__all__ = ['TrainingTask']

class TrainingTask(nn.Module):
    def __init__(self, 
                model: AtomisticModel,
                losses: List[GetLoss],
                device: torch.device = torch.device('cpu'),
                optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_args: Optional[Dict[str, Any]] = None,
                scheduler_cls: Optional[Type] = None,
                scheduler_args: Optional[Dict[str, Any]] = None
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
        
        self.grad_enabled = len(self.model.required_derivatives) > 0

    def forward(self, data):
        data.to(self.device)
        return self.model(data)

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
                print(
                    f"{subset}_{eachloss.name}_{metric_name}",
                    torch.mean(torch.stack(metric))
                )

    def train_step(self, batch):
        batch.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        #self.log_metrics('train', pred, batch)
        return loss.item()

    def validate(self, val_loader):
        torch.set_grad_enabled(self.grad_enabled)
        
        self.eval()
        total_loss = 0
        for batch in val_loader:
            batch.to(self.device)
            pred = self.forward(batch)
            loss = self.loss_fn(pred, batch)
            total_loss += loss.item()
            self.log_metrics('val', pred, batch)
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss
            avg_loss = total_loss / len(train_loader)
            #self.retrieve_metrics('train')

            val_loss = self.validate(val_loader)
            self.retrieve_metrics('val')
            print(f'Epoch {epoch}, Train Loss: {avg_loss}, Val Loss: {val_loss}')

    def save_model(self, path: str):
        torch.save(self.model, path)
