import pandas as pd
import yaml
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm
import os
from pathlib import Path
import re
from shutil import copyfile


def main():
    args = get_args()
    if args.ids_file != '':
        ids = pd.read_csv(args.ids_file, dtype=str).dropna()
        for i, exp in ids.iterrows():
            print(exp['name'])
            # copy_checkpoints(args.base_dir, exp.ids, exp.hparams, exp['name'])
            filename = f'{exp["name"]}_test_rd_curve.csv'
            if not os.path.exists(filename):
                print(f'Generating {filename}')
                create_csv(args.base_dir, exp.ids, args.dataset_pattern, exp.hparams, filename)


def extract_epoch(ckpt_name):
    return int(ckpt_name.split('=')[1].split('-')[0])


def copy_checkpoints(base_dir, experiment_pattern, relevant_hparams, name):
    ckpt_dir = f'{base_dir}/../checkpoints_after_refactoring/{name}/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    experiment_dirs = find_dirs(base_dir, experiment_pattern)
    for experiment_dir in tqdm(experiment_dirs):
        ckpts = glob(experiment_dir + '/checkpoints/epoch=*.ckpt')
        if len(ckpts) == 1:
            ckpt = ckpts[0]
        elif len(ckpts) > 1:
            # print(f'{experiment_dir} has {len(ckpts)} checkpoints')
            ckpt = sorted(ckpts, key=extract_epoch, reverse=True)[0]
            # print(f'Copying {ckpt}')
        else:
            print(f'{experiment_dir} has no checkpoints')
            continue
        hparam_file = experiment_dir + '/tensorboard/hparams.yaml'
        if os.path.exists(hparam_file):
            with open(hparam_file) as f:
                try:
                    hparams = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
                    continue
        hparams = {k: v for k, v in hparams.items() if k in relevant_hparams}
        for k in ['lambda_rd', 'bottleneck_channels']:
            if k in hparams:
                hparams[k] = '-'.join([str(x) for x in hparams[k]])
        target = ckpt_dir + name + '_' + '_'.join([f'{k}={v}' for k, v in hparams.items()]) + '.ckpt'
        copyfile(ckpt, target)




def create_csv(base_dir, experiment_pattern, dataset_pattern, relevant_hparams, name):
    result = pd.DataFrame()
    experiment_dirs = find_dirs(base_dir, experiment_pattern)
    for experiment_dir in tqdm(experiment_dirs):
        hparam_file = experiment_dir + '/tensorboard/hparams.yaml'
        if os.path.exists(hparam_file):
            with open(hparam_file) as f:
                try:
                    hparams = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
                    continue
        hparams = {k: v for k, v in hparams.items() if k in relevant_hparams}
        for k in ['lambda_rd', 'bottleneck_channels']:
            if k in hparams and isinstance(hparams[k], list) and len(hparams[k]) == 1:
                hparams[k] = hparams[k][0]
        dataset_dirs = find_dirs(experiment_dir, dataset_pattern)
        for dataset_dir in dataset_dirs:
            print(dataset_dir)
            hparams['dataset'] = dataset_dir.split('/')[-1]
            for results_file in glob(dataset_dir + '/*.csv'):
                filename = Path(results_file).stem
                if filename != 'rate_distortion':
                    ch_fraction = float('.'.join(filename.split('_')[3:]))
                    hparams['ch_fraction'] = ch_fraction
                    df = pd.read_csv(results_file, index_col=0)
                    avg_results = df.loc['avg']
                    row = pd.DataFrame([dict(**hparams, **avg_results.to_dict())])
                    result = pd.concat([result, row], ignore_index=True)
    if len(result) > 0:
        result.to_csv(name, index=False)


def get_args():
    parser = ArgumentParser(description='''
    Generate rate distortion curves (as csvs) from experiment results.
    The experiments are expected to contain a subdirectory for each dataset they where evaluated on.
    This is the case for trainings that were run with the `--test_sets` argument.
    ''')
    parser.add_argument('--base_dir', type=str, default='../experiments', help='path to dir containing experiments')
    parser.add_argument('--experiment_pattern', type=str, default='.*', help='regex pattern for experiment of interest')
    parser.add_argument('--dataset_pattern', type=str, default='.*val.*', help='regex pattern for dataset of interest')
    parser.add_argument('--relevant_hparams', type=str, nargs='+', default=[], help='list of relevant hparams to include in csv output. All metrics are included by default')
    parser.add_argument('--name', type=str, default='', help='name of the experiment colleciton to be included in csv output')
    parser.add_argument('--ids_file', type=str, default='ids.csv', help='path to file containing experiment ids')
    args = parser.parse_args()
    args.name = '' if not args.name else args.name + '_'
    return args


def find_dirs(base_dir, pattern):
    dirs = list(os.listdir(base_dir))
    matches = list(filter(re.compile(pattern).match, dirs))
    return [base_dir + '/' + match for match in matches]


if __name__ == '__main__':
    main()
