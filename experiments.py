import click
import yaml
from glob import glob
from os import system, path


NOT_USED_CMD = ['training_type_details', 'training_type', 'description']


def read_all_experiments():
    experiments = dict()
    yml_files = glob('./experiments/*yml')
    if len(yml_files) > 0:
        for yml in yml_files:
            with open(yml) as f:
                yaml_loaded = yaml.load(f, Loader=yaml.FullLoader)
                for d in yaml_loaded['experiments']:
                    experiments[d['exp_id']] = d
    if len(experiments) > 0:
        return experiments

    return None


def normal_bin_train(params):
    base = 'python3 cli.py '
    for k, v in params.items():
        if k not in NOT_USED_CMD:
            base += f'--{k} {v} '
    base += 'normal-bin-train '
    for k, v in params['training_type_details'].items():
        if k in ['binary', 'binloss']:
            if v:
                base += f'--{k} '
            else:
                base += f'--no-{k} '

        else:
            base += f'--{k} {v} '
    print(base)
    system(f'{base}')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('exp_ids', type=str)
def train(exp_ids):
    # exp_ids can be just one value, e.g, 102, or multiple values, e.g., 102,103,104,...
    exp_ids = exp_ids.split(',')
    all_experiments = read_all_experiments()
    if all_experiments is not None:
        for e in exp_ids:
            exp = all_experiments[int(e)]

            logdir = f"./output/{e}/"
            if path.exists(logdir):
                exit(f'Exit, because experiment logdir {logdir} already exists')
            normal_bin_train(exp)
    else:
        exit('No experiments were found')


@cli.command()
@click.argument('exp_id', type=int)
def log(exp_id):
    system(f'tensorboard --logdir ./output/{exp_id}/log/ --bind_all')


if __name__ == "__main__":
    cli()
