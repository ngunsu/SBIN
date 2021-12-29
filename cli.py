from utils.train_test_helper import TrainTestHelper
from argparse import Namespace
from normal_bin_template import NormalBinTemplate
from utils.dataloaders.kitti_data_module import KittiDataModule
from utils.dataloaders.sceneflow_data_module import SceneflowDataModule
from pytorch_lightning.trainer import seed_everything
import click
import os

# Definitions
DEFAULT_CPUS = os.cpu_count()
DEFAULT_DATASET_PATH = '/workspace/datasets/'
DATASETS = ['sceneflow', 'kitti2012', 'kitti2015']


@click.group()
@click.option('--num_workers', default=DEFAULT_CPUS, type=int, help="Number of cpus to use")
@click.option('--gpus', default=1, help='Number of gpus to use')
@click.option('--seed', default=1, help='Seed')
@click.option('--shuffle/--no-shuffle', default=True, help="Shuffle data while training")
@click.option('--drop_last/--no-drop_last', default=False, help="Drop last batch during training")
@click.option('--lr', default=5e-3, help='Learning rate')
@click.option('--save_top_k', default=1, help='Save best k models')
@click.option('--batch_size', default=6, help='Batch size')
@click.option('--patience', default=50, help='Early stopping')
@click.option('--scheduler', type=click.Choice(['steplr', 'multisteplr', 'plateau']), default='steplr')
@click.option('--optimizer', type=click.Choice(['adam']), default='adam')
@click.option('--gamma', default=0.1, help='Learning rate step gamma')
@click.option('--gamma_step', default=10, help='Learning rate step')
@click.option('--precision', type=int, default=32, help='Float precision')
@click.option('--max_disp', default=192, help='Maximum disparity')
@click.option('--epochs_per_val', default=1, help='Checks validation every epochs_per_val')
@click.option('--min_epochs', default=10, help='Minimum number of epochs during training')
@click.option('--max_epochs', default=300, help='Maximun number of epochs during training')
@click.option('--dataset', type=click.Choice(DATASETS), default='kitti2012')
@click.option('--datasets_path', default=DEFAULT_DATASET_PATH)
@click.option('--exp_id', default='1', help='Experiment id, used to store checkpoints and log')
@click.option('--debug/--no-debug', default=False, help='Used to test the code with a  small portion of data')
@click.option('--justtest/--no-justtest', default=False, help='Run just the test set')
@click.option('--hyper_search/--no-hyper_search', default=False, help='Hyper search using comet_experiments')
@click.option('--hyper_params', default=None, required=False, type=str, help='Json file with hyper parameters range')
@click.option('--comet_project_name', default='Default_exp')
@click.option('--resume', default=None, required=False, type=str, help='Checkpoint to resume training')
@click.option('--normalization', type=click.Choice(['default', 'imagenet']), default='default')
@click.option('--output_path', default='./output', help='Output folder used to save logs and checkpoints')
@click.option('--save_results/--no-save_results', default=False, help='Save visual results')
@click.option('--results_path', default='./results', help='Output to save results (images)')
@click.option('--pretrained', default='', help='Use pretrained weights in path')
@click.option('--pretrained_student', default='', help='Use pretrained weights from a distilled model')
@click.pass_context
def cli(ctx, **args):
    ctx.obj = args


@cli.command()
@click.option('--model_type', type=str, default='default')
@click.option('--pool', type=click.Choice(['max', 'avg']),
              default='max')
@click.option('--binary/--no-binary', default=False, help='Binary')
@click.option('--binloss/--no-binloss', default=False, help='Add binary loss')
@click.pass_context
def normal_bin_train(ctx, **args):
    all_params = {**ctx.obj, **args}
    hparams = Namespace(**all_params)

    # set seed
    seed_everything(hparams.seed)

    # some hparams used
    hyper_search = hparams.hyper_search

    # select dataset
    dm = KittiDataModule
    if hparams.dataset == 'sceneflow':
        dm = SceneflowDataModule
    nt = NormalBinTemplate
    tthelper = TrainTestHelper(hparams=hparams, pl_module=nt, data_module=dm)

    if not hyper_search:
        tthelper.train()
    else:
        tthelper.hypersearch()


if __name__ == "__main__":
    cli()
