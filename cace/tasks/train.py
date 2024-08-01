from typing import Optional, Dict, List, Type, Any
import logging
import torch
from torch import nn
import numpy as np
from .loss import GetLoss
from ..tools import Metrics
from ..tools import to_numpy, tensor_dict_to_device

"""
This file contains the training loop for the neural network model.
"""

__all__ = ['TrainingTask']

class TrainingTask(nn.Module):
    def __init__(self, 
                model: nn.Module,
                losses: List[GetLoss],
                metrics: List[Metrics],
                device: torch.device = torch.device('cpu'),
                optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_args: Optional[Dict[str, Any]] = None,
                scheduler_cls: Optional[Type] = None,
                scheduler_args: Optional[Dict[str, Any]] = None,
                ema: bool = False, 
                ema_decay: float = 0.99,
                ema_start: int = 0,
                swa: bool = False,
                swa_start: int = 0,
                swa_lr: float = 1e-3,
                swa_losses: List[GetLoss] = [],
                max_grad_norm: float = 10,
                warmup_steps: int = 0,                
                ):
        """
        Args:
            model: the neural network model
            losses: list of losses an optional loss functions
            metrics: list of metrics
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            ema: whether to use exponential moving average
            ema_decay: decay rate of ema
            ema_start: when to start ema
            swa: whether to use stochastic weight averaging
            swa_start: when to start swa
            swa_lr: learning rate for swa
            swa_losses: list of losses for swa
            max_grad_norm: max gradient norm
            warmup_steps: number of warmup steps before reaching the base learning rate
        """

        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.losses = nn.ModuleList(losses)
        self.metrics = nn.ModuleList(metrics)
        self.optimizer = optimizer_cls(self.parameters(), **optimizer_args)
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_args) if scheduler_cls else None
        self.ema = ema
        self.ema_start = ema_start
        if self.ema:
            # AveragedModel is kinda new in torch so there's a fall back
            try:
                self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, \
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay))
            except:
                ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    ema_decay * averaged_model_parameter + (1-ema_decay) * model_parameter
                self.ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        else:
            self.ema_model = None

        self.swa = swa
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_losses_list = swa_losses

        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.lr = optimizer_args['lr']
        self.global_step = 0

        self.grad_enabled = len(self.model.required_derivatives) > 0

    def update_loss(self, losses: List[GetLoss]):
        self.losses = nn.ModuleList(losses)

    def update_metrics(self, metrics: List[Metrics]):
        self.metrics = nn.ModuleList(metrics)

    def forward(self, data, training: bool):
        return self.model(data, training=training)

    def loss_fn(self, pred, batch, loss_args: Optional[Dict[str, torch.Tensor]] = None, index: Optional[List[int]] = None):
        loss = 0.0
        if index is not None:
            for i in index:
                loss += self.losses[i](pred, batch, loss_args)
        else:
            for eachloss in self.losses:
                loss += eachloss(pred, batch, loss_args)
        return loss

    def log_metrics(self, subset, pred, batch):
        for metric in self.metrics:
            metric.update_metrics(subset, pred, batch)

    def retrieve_metrics(self, subset, print_log: bool = False):
        for metric in self.metrics:
            metric.retrieve_metrics(subset, print_log=print_log)

    def train_step(self, 
                   batch, 
                   screen_nan: bool = True, 
                   output_index: Optional[int] = None, # output index for multi-output models
                   loss_index: Optional[List[int]] = None # loss index for multi-loss models
                   ):
        torch.set_grad_enabled(True)

        batch.to(self.device)
        batch_dict = batch.to_dict()

        self.train()
        self.optimizer.zero_grad()
        pred = self.model(batch_dict, training=True, output_index=output_index)
        self.log_metrics('train', pred, batch_dict)

        loss = self.loss_fn(pred, batch_dict, {'epochs': self.global_step, 'training': True}, loss_index)
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
            if self.global_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_scale * self.lr
 
            self.optimizer.step()
            if self.ema and self.global_step >= self.ema_start:
                self.ema_model.update_parameters(self.model)

        return to_numpy(loss).item()

    def validate(self, val_loader, output_index: Optional[int] = None):
        torch.set_grad_enabled(self.grad_enabled)
        
        self.eval()
        total_loss = 0.0
        for batch in val_loader:
            batch.to(self.device)
            batch_dict = batch.to_dict()
            if self.ema and self.global_step >= self.ema_start:
                pred = self.ema_model(batch_dict, training=False, output_index=output_index)
            else:
                pred = self.model(batch_dict, training=False, output_index=output_index)

            loss = to_numpy(self.loss_fn(pred, batch_dict, {'epochs': self.global_step, 'training': False}))
            total_loss += loss.item()
            self.log_metrics('val', pred, batch_dict)

        return total_loss / len(val_loader)

    def fit(self, 
            train_loader, 
            val_loader, 
            epochs, 
            val_stride: int = 1, 
            screen_nan: bool = True,
            checkpoint_path: Optional[str] = 'checkpoint.pt',
            checkpoint_stride: int = 10,            
            bestmodel_path: Optional[str] = 'best_model.pth',
            print_stride: int = 1,
            subset_ratio: float = 1.0,
            output_index: Optional[int] = None, # output index for multi-output models
            subsample_loss_mode: Optional[int] = None,
           ):

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):

            # start SWA if needed
            if self.swa and self.global_step >= self.swa_start:
                if self.swa_model is None:
                    logging.info('SWA started:')
                    if self.ema_model is not None:
                        self.swa_model = torch.optim.swa_utils.AveragedModel(self.ema_model.module)
                    else:
                        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
                    self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=self.swa_lr)
                    if len(self.swa_losses_list) > 0:
                        self.update_loss(self.swa_losses_list)

            # train
            total_loss = 0
            if subset_ratio < 1.0:
                train_loader = self._get_subset_batches(train_loader, subset_ratio)
            for batch in train_loader:
                if subsample_loss_mode is not None:
                    loss_index = np.random.choice(len(self.losses), subsample_loss_mode)
                    loss = self.train_step(batch, screen_nan=screen_nan, loss_index=loss_index, output_index=output_index)
                else:
                    loss = self.train_step(batch, screen_nan=screen_nan, loss_index=None, output_index=output_index)
                total_loss += loss
            avg_loss = total_loss / len(train_loader)

            if self.swa and self.global_step >= self.swa_start:
               self.swa_model.update_parameters(self.model)

            # validate
            if print_stride > 0 and self.global_step % print_stride == 0:
                screen_output = True
            else:
                screen_output = False
            if epoch % val_stride == 0:
                val_loss = self.validate(val_loader, output_index=output_index)
                for pg in self.optimizer.param_groups:
                    lr_now = pg["lr"]
                    if screen_output:
                        print(f"##### Step: {self.global_step} Learning rate: {lr_now} #####")
                    logging.info(f"##### Step: {self.global_step} Learning rate: {lr_now} #####")
                if screen_output:
                    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                logging.info(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                self.retrieve_metrics('train', print_log=screen_output)
                self.retrieve_metrics('val', print_log=screen_output)

            if self.swa and self.global_step >= self.swa_start:
               self.swa_scheduler.step()
            elif self.scheduler:
                if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if epoch > val_stride and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(bestmodel_path, device=self.device)

            if checkpoint_path is not None and epoch % checkpoint_stride == 0:
                self.checkpoint(checkpoint_path)

            self.global_step += 1 

    # Function to subsample batches
    def _get_subset_batches(self, dataloader, subset_ratio: float):
        batches = list(dataloader)
        subset_size = int(subset_ratio * len(dataloader))
        if subset_size == 0: subset_size = 1
        indices = np.random.choice(len(batches), subset_size, replace=False)
        return [batches[i] for i in indices]

    def save_model(self, path: str, device: torch.device = torch.device('cpu')):
        if self.ema and self.global_step >= self.ema_start: 
            torch.save(self.ema_model.module.to(device), path)
            if device != self.device:
                self.ema_model.module.to(self.device)
        else:
            torch.save(self.model.to(device), path)
            if device != self.device:
                self.model.to(self.device)

    def checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_ema_state_dict': self.ema_model.state_dict() if self.ema and self.global_step >= self.ema_start else None,
            'model_swa_state_dict': self.swa_model.state_dict() if self.swa and self.global_step >= self.swa_start else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        if self.ema:
            self.ema_model.load_state_dict(state_dict['model_ema_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


