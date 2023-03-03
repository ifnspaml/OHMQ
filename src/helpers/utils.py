import os
import torch
import numpy as np
import random
from torchvision import transforms
from contextlib import contextmanager
from pytorch_lightning import Trainer
import signal
from pathlib import Path
from functools import partial
import sys
from typing import Optional


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_job_id(dir_for_running_local_id):
    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID', None)
    array_task_id = os.getenv('SLURM_ARRAY_TASK_ID', None)
    if (array_job_id is not None) and (array_task_id is not None):
        job_id = f'{array_job_id}_{array_task_id}'
    else:
        job_id = os.getenv('SLURM_JOB_ID', default=get_next_id(dir_for_running_local_id))
    return job_id


def setup_generic_signature(args):
    job_id = get_job_id('experiments/')
    args.name = job_id
    if args.description:
        args.name += '_' + args.description

    print(f"Experiment name: {args.name}")
    args.job_id = job_id
    if args.output_dir is None:
        args.snapshot = os.path.join('experiments', args.name)
    else:
        args.snapshot = args.output_dir
    args.checkpoints_save = os.path.join(args.snapshot, 'checkpoints')
    args.storage_save = os.path.join(args.snapshot, 'storage')
    args.tensorboard_runs = os.path.join(args.snapshot, 'tensorboard')
    args.inference_dir = os.path.join(args.snapshot, 'inference')

    makedirs(args.snapshot)
    makedirs(args.checkpoints_save)
    makedirs(args.storage_save)

    return args


def make_deterministic_train(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)


def inverse_normalization(openimages=False):
    if openimages:
        return transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    else:
        return transforms.Normalize(mean=-1, std=2)


def save_checkpoint_on_sigterm(trainer: Trainer):
    def save_ckpt_handler(sig_number, frame, old_handler=None):
        print(f'received signal {sig_number}. Saving checkpoint')
        filepath = trainer.checkpoint_callback.dirpath + '/on_signal.ckpt'
        trainer.save_checkpoint(filepath)
        if old_handler is None:
            sys.exit(1)
        else:
            return old_handler(sig_number, frame)
    try:
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGCONT]
    except:
        signals = [signal.SIGINT, signal.SIGTERM]
    for s in signals:
        signal.signal(s, partial(save_ckpt_handler, old_handler=signal.getsignal(s)))


def find_ckpt(optional_checkpoint: Optional[str], job_id: str=None):
    base_dir = 'experiments/'
    ckpt = None
    if optional_checkpoint is None:
         ckpt = find_ckpt_with_current_id(job_id, base_dir)
    elif optional_checkpoint.endswith('.ckpt'):
        ckpt = optional_checkpoint
    elif optional_checkpoint[:6].isnumeric():
        experiment_dirs = list(Path(base_dir).glob(f'{optional_checkpoint}*/'))
        if len(experiment_dirs) != 1:
            print(f'could not find experiment_dir for id {optional_checkpoint}')
        else:
            experiment_dir = experiment_dirs[0]
            ckpts = list(Path(experiment_dir).glob('checkpoints/epoch*.ckpt'))
            if len(ckpts) == 1:
                ckpt = ckpts[0]
            else:
                print(f'possible early stopping checkpoints to load:\n{ckpts}')
                ckpts = list(Path(experiment_dir).glob('epoch=last.ckpt'))
                if len(ckpts) == 1:
                    ckpt = ckpts[0]
    return ckpt


def find_ckpt_with_current_id(job_id, experiment_dir='experiments/'):
    result_list = list(Path(experiment_dir).glob(f'{job_id}*/**/on_signal.ckpt'))
    return result_list[0] if result_list else None


def get_next_id(experiment_dir):
    job_id = 0
    while id_exists(job_id, experiment_dir):
        job_id += 1
    return str(job_id)


def id_exists(job_id, experiment_dir):
    found_paths = Path(experiment_dir).glob(f'{job_id}*/')
    return next(found_paths, None) is not None


class TemproaryPadding:
    def __init__(self, down_factor, img_height, img_width, warn=True):
        height, width = [(down_factor - s % down_factor) % down_factor for s in (img_height, img_width)]
        self.top, self.left = [total_pad_size // 2 for total_pad_size in (height, width)]
        self.bottom, self.right = [(total_pad_size + 1) // 2 for total_pad_size in (height, width)]
        self.padding = torch.nn.ZeroPad2d((self.left, self.right, self.top, self.bottom))
        self.has_effect = (height + width) > 0
        self.warn = warn

    def apply(self, x: torch.Tensor):
        if self.has_effect:
            x = self.padding(x)
            if self.warn:
                print('padded image to suit network architecture')
        return x

    def unpad(self, x: torch.Tensor):
        x = x[:, :, self.top:, self.left:]
        if self.bottom > 0:
            x = x[:, :, :-self.bottom, :]
        if self.right > 0:
            x = x[:, :, :, :-self.right]
        return x


def ste(forward_val, backward_val):
    return forward_val.detach() + backward_val - backward_val.detach()


def calc_num_channels(channel_fraction, min_channels, max_channels):
    if isinstance(channel_fraction, (int, float)):
        channel_fraction = torch.tensor([channel_fraction], dtype=torch.float32)
    return min_channels + torch.ceil(channel_fraction * (max_channels - min_channels)).int()

