import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
from typing import Dict, Optional, List, Tuple

#Standard class for wrapping around 
class LightningModel(L.LightningModule):
    def __init__(self,
                 model : nn.Module,
                 losses : List[nn.Module] = None,
                 metrics : List[nn.Module] = None,
                 metric_labels : List[str] = ["e","f"],
                 log_rmse : bool = True,
                 optimizer_args = {'lr': 1e-2},
                 train_args = {"training":True},
                 val_args = {"training":False},
                 pass_epoch_to_loss = False,
                ):
        super().__init__()
        from deeporb.util import default_losses, default_metrics
        if losses is None:
            losses = default_losses()
        if metrics is None:
            metrics = default_metrics()
        self.model = model
        self.losses = losses
        self.metrics = metrics
        self.metric_labels = metric_labels
        self.log_rmse = log_rmse
        self.optimizer_args = optimizer_args
        self.train_args = train_args
        self.val_args = val_args
        self.pass_epoch_to_loss = pass_epoch_to_loss

    def forward(self,
                data: Dict[str, torch.Tensor],
                kwargs = None, #dict
               ) -> Dict[str, torch.Tensor]:
        if kwargs is None:
            kwargs = self.val_args
        return self.model.forward(data,**kwargs)

    def calculate_loss(self, 
            data: Dict[str, torch.Tensor], 
            ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        
        #Forward -- get a dictionary of predicted tensors
        results = self.forward(data,self.train_args)
        
        #Calculate loss
        epoch = self.trainer.current_epoch
        loss_args = None
        if self.pass_epoch_to_loss:
            loss_args = {"epoch":epoch}
        tot_loss = 0
        for i, loss_fn in enumerate(self.losses):
            loss = loss_fn(results,data,loss_args=loss_args)
            tot_loss = tot_loss + loss
        return tot_loss, results
        
    def calculate_metrics(self, 
                data: Dict[str, torch.Tensor],
                results : Dict[str, torch.Tensor],
                ) -> Tuple[Dict[int, torch.Tensor], str]:
        dct = {}
        typ = "rmse"
        if not self.log_rmse:
            typ = "mae"
        for i,metric in enumerate(self.metrics):
            name = self.metric_labels[i]
            dct[name] = metric(results,data)[typ]
        return dct, typ

    def training_step(self,
                      data : Dict[int, torch.Tensor],
                      batch_idx : int) -> torch.Tensor:
        loss, results = self.calculate_loss(data)

        #Calc metrics
        batch_size = data.batch.max() + 1
        dct, typ = self.calculate_metrics(data,results)
        for k,v in dct.items():
            self.log(f"train_{k}_{typ}",v,batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_args)
        return optimizer

    def validation_step(self,
                      data : Dict[int, torch.Tensor],
                      val_idx : int) -> None:
        batch_size = data.batch.max() + 1

        #Get data
        with torch.enable_grad(): #for forces
            results = self.forward(data,self.val_args)

        #Log metrics
        dct, typ = self.calculate_metrics(data,results)
        for k,v in dct.items():
            self.log(f"val_{k}_{typ}",v,batch_size=batch_size)

class LightningTrainingTask():
    def __init__(self,
                 model : nn.Module,
                 losses : List[nn.Module] = None,
                 metrics : List[nn.Module] = None,
                 metric_labels : List[str] = ["e","f"],
                 log_rmse : bool = True,
                 optimizer_args = {'lr': 1e-2},
                 train_args = {"training":True},
                 val_args = {"training":False},
                 pass_epoch_to_loss = False,
                ) -> None:
        self.model = LightningModel(model,
                                    losses = losses,
                                    metrics = metrics,
                                    metric_labels = metric_labels,
                                    log_rmse = log_rmse,
                                    optimizer_args = optimizer_args,
                                    train_args = train_args,
                                    val_args = val_args,
                                    pass_epoch_to_loss = pass_epoch_to_loss,
        )

    def fit(self,data,chkpt=None,dev_run=False,max_epochs=100,gradient_clip_val=10):
        if chkpt is not None:
            self.load(chkpt)
        trainer = L.Trainer(fast_dev_run=dev_run,max_epochs=max_epochs,gradient_clip_val=gradient_clip_val)
        trainer.fit(self.model,data,ckpt_path=chkpt)

    def save(self,path):
        print("Saving model to",path,"...")
        state_dict = self.model.state_dict()
        torch.save({"state_dict":state_dict}, path)

    def load(self,path):
        print("Loading model from",path,"...")
        state_dict = torch.load(path, weights_only=True)
        self.model.load_state_dict(state_dict["state_dict"])
        print("Loading successful!")


#Data
from cace.cace.tools import torch_geometric
from cace.cace.data import AtomicData
from cace.cace.tasks import get_dataset_from_xyz, load_data_loader

class LightningData(L.LightningDataModule):
    def __init__(self, root,
                 cutoff=5.5,
                 batch_size=4,
                 data_key = {"energy":"energy","force":"force"},
                 atomic_energies=None,
                 valid_p=0.1,
                 seed=1,
                ):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.valid_p = valid_p
        self.cutoff = cutoff
        self.seed = seed
        self.atomic_energies = atomic_energies
        self.data_key = data_key
        self.prepare_data()
    
    def prepare_data(self):
        collection = get_dataset_from_xyz(train_path=self.root,
                                          valid_fraction=self.valid_p,
                                          seed=self.seed,
                                          cutoff=self.cutoff,
                                          data_key=self.data_key,
                                          atomic_energies = self.atomic_energies
                                         )
        self.collection = collection
        
        self.train_loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=self.cutoff, data_key=self.data_key, atomic_energies=self.atomic_energies)
                for atoms in collection.train
            ],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count()-1,
        )
        
        self.valid_loader = torch_geometric.DataLoader(
            dataset=[
                AtomicData.from_atoms(atoms, cutoff=self.cutoff, data_key=self.data_key, atomic_energies=self.atomic_energies)
                for atoms in collection.valid
            ],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=os.cpu_count()-1,
        )

    def train_dataloader(self):
        return self.train_loader
        
    def val_dataloader(self):
        return self.valid_loader