import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#from api_neurotask import *

import random

from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from IPython.display import Video

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import pytorch_lightning as lightning

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import xfads.utils as utils
import xfads.prob_utils as prob_utils

from xfads import plot_utils

from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningMonkeyReaching
from xfads.smoothers.nonlinear_smoother_causal import NonlinearFilter, LowRankNonlinearStateSpaceModel

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_HOME'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1'

torch.cuda.empty_cache()



def main():
    
    """config"""

    cfg = {
        # --- graphical model --- #
        'n_latents': 40,
        'n_latents_read': 35,

        'rank_local': 15,
        'rank_backward': 5,

        'n_hidden_dynamics': 128,

        # --- inference network --- #
        'n_samples': 25,
        'n_hidden_local': 256,
        'n_hidden_backward': 128,

        # --- hyperparameters --- #
        'use_cd': False,
        'p_mask_a': 0.0,
        'p_mask_b': 0.0,
        'p_mask_apb': 0.0,
        'p_mask_y_in': 0.0,
        'p_local_dropout': 0.4,
        'p_backward_dropout': 0.0,

        # --- training --- #
        'device': 'cuda',
        'data_device': 'cpu',

        'lr': 1e-3,
        'n_epochs': 1000,
        'batch_sz': 512,

        # --- misc --- #
        'bin_sz': 5e-3,
        'bin_sz_ms': 5,

        'n_bins_bhv': 100,

        'go_cue': 152,

        'seed': 1236,
        'default_dtype': torch.float32,

        # --- ray --- #
        'n_ray_samples': 10,
    }


    class Cfg(dict):
        def __getattr__(self, attr):
            if attr in self:
                return self[attr]
            else:
                raise AttributeError(f"'DictAsAttributes' object has no attribute '{attr}'")


    cfg = Cfg(cfg)

    torch.cuda.empty_cache()

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)


    """"Loading data"""

    data_path = 'data/data_{split}_{bin_size_ms}ms.pt'
    train_data = torch.load(data_path.format(split='train', bin_size_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split='valid', bin_size_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split='test', bin_size_ms=cfg.bin_sz_ms))

    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)
    y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)

    label_train = train_data['label'].type(torch.float32).to(cfg.data_device)[:, :, [2, 3]]
    label_valid = val_data['label'].type(torch.float32).to(cfg.data_device)[:, :, [2, 3]]
    label_test = test_data['label'].type(torch.float32).to(cfg.data_device)[:, :, [2, 3]]

    train_moveonset = train_data['move_onset'].type(torch.float32).to(cfg.data_device)
    valid_moveonset = val_data['move_onset'].type(torch.float32).to(cfg.data_device)
    test_moveonset = test_data['move_onset'].type(torch.float32).to(cfg.data_device)

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, label_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, label_valid)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, label_test)

    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_test_obs.shape[0], shuffle=False)

    # Data dimensions
    n_train_trials, n_time_bins, n_neurons_obs = y_train_obs.shape
    n_valid_trials = y_valid_obs.shape[0]
    n_test_trials = y_test_obs.shape[0]
    n_time_bins_enc = train_data['n_time_bins_enc']

    batch_sz_train = list(y_train_obs.shape)[:-1]
    batch_sz_valid = list(y_valid_obs.shape)[:-1]
    batch_sz_test = list(y_test_obs.shape)[:-1]

    print("# training trials: {0}".format(n_train_trials))
    print("# validation trials: {0}".format(n_valid_trials))
    print("# testing trials: {0}".format(n_test_trials))
    print("# neurons: {0}".format(n_neurons_obs))
    print("# time bins: {0}".format(n_time_bins))
    print("# time bins used for forcasting: {0}".format(cfg.n_bins_bhv))
    print("# predicted time bins: {0}".format(n_time_bins - cfg.n_bins_bhv))


    """likelihood pdf"""
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
    readout_fn[-1].bias.data = prob_utils.estimate_poisson_rate_bias(train_dataloader, cfg.bin_sz)
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents,
                                      rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)

    #import multiprocessing as mp
    #mp.set_start_method('spawn')

    seq_vae = LightningMonkeyReaching(ssm, cfg, n_time_bins_enc, cfg.n_bins_bhv)
    csv_logger = CSVLogger('logs/smoother/acausal/', name=f'sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}', version='smoother_acausal')
    ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='r2_valid_enc', mode='max', dirpath='ckpts/smoother/acausal/', save_last=True,
                                    filename='{epoch:0}_{valid_loss:0.2f}_{r2_valid_enc:0.2f}_{r2_valid_bhv:0.2f}_{valid_bps_enc:0.2f}')

    trainer = lightning.Trainer(max_epochs=cfg.n_epochs,
                                #accelerator='gpu',
                                #devices='auto',
                                #strategy='deepspeed_stage_1', precision=16,
                                #strategy='ddp', # raise RuntimeError("Distributed package doesn't have NCCL built in") RuntimeError: Distributed package doesn't have NCCL built in
                                gradient_clip_val=1.0,
                                default_root_dir='lightning/',
                                callbacks=[ckpt_callback],
                                logger=csv_logger,
                                )

    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    torch.save(ckpt_callback.best_model_path, 'ckpts/smoother/acausal/best_model_path.pt')
    trainer.test(dataloaders=test_dataloader, ckpt_path='last')


if __name__ == '__main__':
    main()