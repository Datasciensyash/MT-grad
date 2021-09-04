import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
import typing as tp


class LightningEngine(pl.LightningModule):

    def __init__(
            self,
            config: dict,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.StepLR,
            criterion: nn.Module,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.save_hyperparameters('config')

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(y_pred, y_true)

    def _compute_metrics(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, prefix: str = 'train'
    ) -> tp.Dict[str, float]:
        l1_distance = torch.abs(y_true - y_pred).mean()
        return {f'{prefix}/l1_distance': l1_distance}

    def training_step(self, batch, batch_idx):
        (rho_yx, rho_xy), (phi_yx, phi_xy), target = batch
        pred_labels = self.model(rho_yx, rho_xy, phi_yx, phi_xy)
        loss = self._compute_loss(pred_labels, target)
        metrics = self._compute_metrics(pred_labels, target, prefix='train')
        self.log('train/loss', loss.cpu().item())
        self.log_dict(metrics)
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch, batch_idx):
        (rho_yx, rho_xy), (phi_yx, phi_xy), target = batch
        pred_labels = self.model(rho_yx, rho_xy, phi_yx, phi_xy)
        loss = self._compute_loss(pred_labels, target)
        metrics = self._compute_metrics(pred_labels, target, prefix='valid')
        self.log('valid/loss', loss.cpu().item())
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        (rho_yx, rho_xy), (phi_yx, phi_xy), target = batch
        pred_labels = self.model(rho_yx, rho_xy, phi_yx, phi_xy)
        loss = self._compute_loss(pred_labels, target)
        metrics = self._compute_metrics(pred_labels, target, prefix='test')
        self.log('valid/loss', loss.cpu().item())
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
