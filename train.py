import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pytorch_lightning.loggers import TensorBoardLogger
from config import args as default_args
from src.helpers import utils
from src.systems import MnistAutoencoder, OpenimagesAutoencoder
import warnings
from compress import compress
from src.datasets import prepare_dataloaders
warnings.filterwarnings("ignore")


def train(args):
    utils.make_deterministic_train(args.seed)
    train_loader, val_loader = prepare_dataloaders(args)
    system = (MnistAutoencoder if args.mnist else OpenimagesAutoencoder)(**vars(args))
    checkpointer = ModelCheckpoint(monitor='val_loss', dirpath=args.checkpoints_save, save_last=True)
    logger = TensorBoardLogger(save_dir=args.tensorboard_runs, name='', version='', log_graph=False)
    logger.log_hyperparams(system.hparams)
    ckpt = utils.find_ckpt(args.ckpt, args.job_id)
    gpus = 1 if torch.cuda.is_available() and not args.cpu else 0
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpointer],
        default_root_dir=args.snapshot,
        logger=logger,
        max_epochs=args.n_epochs,
        gpus=gpus,
        log_every_n_steps=100,
        resume_from_checkpoint=ckpt,
    )
    trainer.fit(system, train_loader, val_loader)
    ckpt = checkpointer.best_model_path
    results = None
    for test_set in args.test_sets:
        if 'openimages/validation' not in test_set:  # this would take forever
            output_dir = f'{args.inference_dir}_{test_set.replace("/", "_")}'
            results = compress(ckpt, test_set, output_dir=output_dir)
            print(f'results for {test_set}\n{results}')
    return results


def override_defaults_with_cmd_args(cmd_args):
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(default_args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args)
    return args


def get_argument_parser():
    parser = ArgumentParser(description="Training of learnable compression.", formatter_class=ArgumentDefaultsHelpFormatter)
    general = parser.add_argument_group('General options')
    general.add_argument('-log_interval', type=int, default=default_args.log_interval)
    general.add_argument('-seed', type=int, default=default_args.seed)
    general.add_argument('--mnist', action="store_true", help='use mnist models and dataset')
    general.add_argument('-n_epochs', type=int, default=default_args.n_epochs, help='number of epochs to train')
    general.add_argument('--output_dir', type=str, default=None, help='output directory for traning artifacts')

    # dataset
    general.add_argument('--data_base_path', type=str, default=default_args.data_base_path, help='path to datasets')
    general.add_argument('-num_train_samples', type=int, default=default_args.num_train_samples, help='use only a subset of x training images (x=-1 for using all images)')
    general.add_argument('-num_val_samples', type=int, default=default_args.num_val_samples, help='use only a subset of x validation images (x=-1 for using all images)')
    general.add_argument('-crop_size', type=int, default=default_args.crop_size, help='crop size for training and validation (not applied to mnist models)')
    general.add_argument('-batch_size', type=int, default=default_args.batch_size, help="Input batch size for training")
    general.add_argument('--description', type=str, default='', help="description to be added to the run name")
    general.add_argument('--num_workers', type=int, default=default_args.num_workers, help="number of processes for data loading")
    general.add_argument('-cpu', action="store_true", help='use cpu even if cuda is available')
    general.add_argument('--ckpt', type=str, default=None, help='path to a checkpoint to resume training from')
    general.add_argument('--test_sets', nargs='+', type=str, default=[], help='list of test sets to run inference on after training')

    parser = pl.Trainer.add_argparse_args(parser)
    known_args, _ = parser.parse_known_args()
    parser = MnistAutoencoder if known_args.mnist else OpenimagesAutoencoder
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    cmd_args = parser.parse_args()
    args = override_defaults_with_cmd_args(cmd_args)
    train(args)

