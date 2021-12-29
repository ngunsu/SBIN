import torch
import os
import time
import numpy as np
import cv2
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from utils.pl.comet import CometLogger
from utils.metrics.metrics import compute_epe, compute_err


class TrainTestHelper():

    """ Train and test helper """

    def __init__(self, hparams, pl_module, data_module):
        """Constructor

        Parameters
        ----------
            hparams (Namespace): Parameters list
            pl_module (LightningModule): PytorchLightning LightningModule
            data_module (LightningDataModule): PytorchLightning LightningDataModule

        Returns
        -------
            None
        """
        # Set arguments
        self.hparams = hparams
        self.lightning_module = pl_module
        self.lightning_data_module = data_module
        self.comet_experiment = None

    def train(self):
        """Train the network"""

        hparams = self.hparams
        dataset = hparams.dataset

        self.pl_module = self.lightning_module(hparams)
        self.data_module = self.lightning_data_module(hparams)
        self.data_module.setup()

        exp_id = hash(time.time())
        if hparams.exp_id != '0':
            exp_id = hparams.exp_id

        # Checkpoint callback
        checkpoint_callback = False
        callbacks = []
        if not hparams.justtest:
            checkpoint_callback = True
            filepath = f'{hparams.output_path}/{exp_id}/ckpt'
            checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                                  save_top_k=hparams.save_top_k,
                                                  verbose=True,
                                                  monitor='val_loss',
                                                  mode='min',
                                                  prefix=f'{dataset}')
            callbacks.append(checkpoint_callback)

        logger = False
        if not hparams.justtest:
            if hparams.hyper_search:
                logger = CometLogger(api_key=os.environ['COMET_KEY'],
                                     project_name=hparams.comet_project_name,
                                     rest_api_key=os.environ["COMET_REST_KEY"],
                                     experiment_name=exp_id,
                                     experiment=self.comet_experiment)
            else:
                logger_path = f'{hparams.output_path}/{exp_id}/log'
                logger = TensorBoardLogger(logger_path)

        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.00,
                                            patience=hparams.patience,
                                            verbose=True,
                                            mode='min')
        callbacks.append(early_stop_callback)

        # Set trainer
        trainer = Trainer(gpus=hparams.gpus,
                          checkpoint_callback=checkpoint_callback,
                          precision=hparams.precision,
                          deterministic=True,
                          fast_dev_run=hparams.debug,
                          check_val_every_n_epoch=hparams.epochs_per_val,
                          min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          resume_from_checkpoint=hparams.resume,
                          progress_bar_refresh_rate=1 if not hparams.hyper_search else 0,
                          logger=logger,
                          callbacks=callbacks)

        if hparams.pretrained != '':
            checkpoint = torch.load(hparams.pretrained)
            self.pl_module.load_state_dict(checkpoint['state_dict'])

        if not hparams.justtest:
            trainer.fit(self.pl_module, self.data_module)
            best_val = early_stop_callback.best_score.float().item()
            print(f'Best val {early_stop_callback.best_score}')
            trainer.test(self.pl_module, datamodule=self.data_module)
            return best_val
        else:
            trainer.test(self.pl_module, datamodule=self.data_module)

        # Save images
        if self.hparams.save_results:
            exp_id = self.hparams.pretrained.split('/')[-2]
            result_folder = os.path.join(hparams.results_path, exp_id)
            os.system(f'mkdir -p {result_folder}')
            self.pl_module.eval()
            file_object = open(os.path.join(result_folder, 'metrics'), "w")
            for idx, batch in enumerate(self.data_module.test_dataset):
                im_left, im_right, d_gt = batch
                im_left = im_left.unsqueeze(0).cuda().contiguous()
                im_right = im_right.unsqueeze(0).cuda().contiguous()
                d_gt = d_gt.squeeze(0)
                d_est = self.pl_module.forward(im_left, im_right)
                if len(d_est) > 1:
                    d_est, _ = d_est

                # Store results metrics
                epe = compute_epe(torch.Tensor(d_gt).cuda(), d_est.squeeze(0).squeeze(0), max_disp=192)
                err = compute_err(torch.Tensor(d_gt).cuda(), d_est.squeeze(0).squeeze(0), max_disp=192, tau=3)
                file_object.write(f'Image {idx} epe: {epe} err3: {err}\n')

                # Prepare for save images
                max_val = d_gt.astype(np.uint8).max()
                d_est = d_est.detach().squeeze(0).squeeze(0).cpu().numpy()
                d_diff = np.abs(d_est - d_gt)
                d_diff[d_gt == 0] = 0
                d_diff[d_diff > 5] = 255
                d_diff[(d_diff > 4) & (d_diff < 240)] = 200
                d_diff[(d_diff > 3) & (d_diff < 190)] = 150
                d_diff[(d_diff > 2) & (d_diff < 140)] = 100
                d_diff[(d_diff > 1) & (d_diff < 100)] = 50
                d_est_abs = cv2.convertScaleAbs(d_est.astype(np.uint8), alpha=255 / max_val)
                d_est_color = cv2.applyColorMap(d_est_abs, cv2.COLORMAP_JET)
                d_gt_abs = cv2.convertScaleAbs(d_gt.astype(np.uint8), alpha=255 / max_val)
                d_gt_color = cv2.applyColorMap(d_gt_abs, cv2.COLORMAP_JET)
                d_diff_color = cv2.applyColorMap(d_diff.astype(np.uint8), cv2.COLORMAP_JET)
                # kd_diff_color = cv2.cvtColor(d_diff, cv2.COLOR_GRAY2BGR)
                stacked_im = np.hstack((d_est_color, d_gt_color, d_diff_color))
                cv2.imwrite(os.path.join(result_folder, f'{idx}.png'), stacked_im)
            file_object.close()

    def hypersearch(self):
        """ Perform hypersearch using comet
        COMET_KEY env variable must be set to work
        """
        from comet_ml import Optimizer
        opt = Optimizer(self.hparams.hyper_params, api_key=os.environ["COMET_KEY"])
        project_name = self.hparams.comet_project_name
        for experiment in opt.get_experiments(project_name=project_name):
            comet_key = experiment.get_key()
            for key in opt.status()['parameters'].keys():
                setattr(self.hparams, key, experiment.get_parameter(key))

            self.hparams.exp_id = comet_key
            self.comet_experiment = experiment

            best_val = self.train()
            experiment.log_metric("best_val", best_val)
