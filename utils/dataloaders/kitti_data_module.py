import pytorch_lightning as pl
from utils.dataloaders.kitti import KittiLoader
from torchvision import transforms
from os.path import join
from torch.utils.data import random_split, DataLoader


class KittiDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams (Namespace): Dataset parameters

        Returns
        -------
        None
        """
        super().__init__()
        self.hparams = hparams

    # -------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------
    def prepare_data(self):
        """ Not needed, just to remember that is possible to implement """
        pass

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    def setup(self):
        # Prepare transform
        if self.hparams.normalization == 'default':
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        elif self.hparams.normalization == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise NameError(f' Normalization {self.normalization} not supported')

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        dataset_path = join(self.hparams.datasets_path, self.hparams.dataset)

        self.full_train_loader = KittiLoader(dataset=self.hparams.dataset,
                                             dataset_path=dataset_path,
                                             training=True,
                                             validation=False,
                                             transform=transform)
        self.test_dataset = KittiLoader(dataset=self.hparams.dataset,
                                        dataset_path=dataset_path,
                                        training=False,
                                        validation=True,
                                        transform=transform)

        train_size = int(len(self.full_train_loader) * 0.9)
        lengths = [train_size, len(self.full_train_loader) - train_size]
        self.train_dataset, self.val_dataset = random_split(self.full_train_loader, lengths)

    # -------------------------------------------------------------------
    # PL dataloaders
    # -------------------------------------------------------------------
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle=self.hparams.shuffle,
                            num_workers=self.hparams.num_workers,
                            drop_last=self.hparams.drop_last)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_workers,
                            drop_last=False)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=self.hparams.num_workers,
                            drop_last=False)
        return loader
