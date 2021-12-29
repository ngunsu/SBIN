import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pytorch_lightning as pl
from model.model_bin_factory import model_bin_factory
from model.bin.binloss import BinLoss
from utils.metrics.metrics import compute_epe, compute_err


class NormalBinTemplate(pl.LightningModule):

    # -------------------------------------------------------------------
    # Training details - Network definition
    # -------------------------------------------------------------------
    def __init__(self, hparams):
        """ Constructor

        Params
        ------
        hparams:
            Contains the training configuration details
        """
        super().__init__()
        self.hparams = hparams
        self.net = model_bin_factory(hparams.model_type, self.hparams.binary, hparams.max_disp)

        if self.hparams.binloss:
            self.binloss = BinLoss()

    # -------------------------------------------------------------------
    # Training details - Optimizer
    # -------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.hparams.lr
        gamma_step = self.hparams.gamma_step
        gamma = self.hparams.gamma

        # Optimizer
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        else:
            raise NameError(f'Optimizer {self.hparams.optimizer} not supported')

        # Scheduler
        if self.hparams.scheduler == 'steplr':
            scheduler = StepLR(optimizer, step_size=gamma_step, gamma=gamma)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif self.hparams.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=25, factor=gamma)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        else:
            return {'optimizer': optimizer}

    # -------------------------------------------------------------------
    # Training details - Forward
    # -------------------------------------------------------------------
    def forward(self, left, right):
        if self.net.training:
            disp, feats, cost, cost_post = self.net.forward(left, right)
        else:
            disp = self.net.forward(left, right)
        return disp

    # -------------------------------------------------------------------
    # Training details - Train step
    # -------------------------------------------------------------------
    def training_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        if self.hparams.dataset == 'sceneflow':
            mask = disp_l < self.hparams.max_disp
        else:
            mask = disp_l > 0
        mask.detach_()

        loss = torch.nn.functional.smooth_l1_loss(output[mask], disp_l[mask], size_average=True)

        if self.hparams.binloss:
            binloss = self.binloss.forward(self.net)
            loss = loss + binloss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # -------------------------------------------------------------------
    # Training details - Validation step
    # -------------------------------------------------------------------
    def validation_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        epe = compute_epe(disp_l, output, max_disp=self.hparams.max_disp)
        self.log('val_loss', epe)

    # -------------------------------------------------------------------
    # Test details - Test step
    # -------------------------------------------------------------------
    def test_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        epe = compute_epe(disp_l, output, max_disp=self.hparams.max_disp)
        err2 = compute_err(disp_l, output, tau=2)
        err3 = compute_err(disp_l, output, tau=3)
        err4 = compute_err(disp_l, output, tau=4)
        err5 = compute_err(disp_l, output, tau=5)

        self.log('test_epe', epe, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('err2', err2, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('err3', err3, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('err4', err4, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('err5', err5, on_step=False, on_epoch=True, prog_bar=False, logger=True)
