# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Modified by: craig.starr
# ------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ------------------------------------------------------------------------------------

import json
import logging
import os
import pprint
import sys
from typing import Any, List

import torch
from omegaconf import DictConfig, OmegaConf

from misc.attr_dict import AttrDict


def save_config_to_disk(cfg):
    filename = os.path.join(cfg.CKPT_PATH, cfg.EXPR_NAME, "config.json")
    with open(filename, "w") as fopen:
        json.dump(cfg, fopen, indent=4, ensure_ascii=False)
        fopen.flush()
    logging.info(f"Saved Config Data to File: {filename}")


def is_hydra_available():
    """
    Check if Hydra is available. Simply python import to test.
    """
    try:
        import hydra  # NOQA

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    logging.getLogger().setLevel(logging.DEBUG)
    if isinstance(cfg, DictConfig):
        logging.info(cfg.pretty())
    else:
        logging.info(pprint.pformat(cfg))


def initialize_config(cfg: DictConfig, mode: str, cmdline_args: List[Any] = None):
    if cmdline_args:
        # convert the command line args to DictConfig
        sys.argv = cmdline_args
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # assert the config and infer
    cfg = cfg.config
    cfg.MODE = mode
    cfg = infer_and_assert_hydra_config(cfg)
    return cfg


def infer_and_assert_hydra_config(cfg):
    # inference
    if cfg.MODE == "extract_shot":
        cfg.TRAIN.DROP_LAST = False
        cfg.TRAIN.SHUFFLE = False
        assert cfg.DISTRIBUTED.NUM_NODES == 1
        cfg.DISTRIBUTED.NUM_PROC_PER_NODE = (
            torch.cuda.device_count()
        )  # use all GPUs available
        assert len(cfg.LOAD_FROM) > 0

    # distributed
    cfg.DISTRIBUTED.WORLD_SIZE = int(
        cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    )
    assert cfg.TRAIN.BATCH_SIZE.effective_batch_size % cfg.DISTRIBUTED.WORLD_SIZE == 0
    cfg.TRAIN.BATCH_SIZE.batch_size_per_proc = int(
        cfg.TRAIN.BATCH_SIZE.effective_batch_size / cfg.DISTRIBUTED.WORLD_SIZE
    )
    cfg.TRAINER.gpus = cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    cfg.TRAINER.num_nodes = cfg.DISTRIBUTED.NUM_NODES
    if cfg.MODEL.use_sync_bn:
        cfg.TRAINER.sync_batchnorm = True

    # auto scale learning rate
    cfg.TRAIN.OPTIMIZER.lr.scaled_lr = cfg.TRAIN.OPTIMIZER.lr.base_lr
    if cfg.TRAIN.OPTIMIZER.lr.auto_scale:
        cfg.TRAIN.OPTIMIZER.lr.scaled_lr = (
            cfg.TRAIN.OPTIMIZER.lr.base_lr
            * cfg.TRAIN.BATCH_SIZE.effective_batch_size
            / float(cfg.TRAIN.OPTIMIZER.lr.base_lr_batch_size)
        )

    # logging
    if cfg.TEST.KNN_VALIDATION:
        # cfg.TRAINER.num_sanity_val_steps = -1
        cfg.TRAINER.check_val_every_n_epoch = cfg.TEST.VAL_FREQ

    # set paths
    assert "PYTHONPATH" in os.environ
    # cfg.PROJ_ROOT = os.environ["PYTHONPATH"]
    cfg.PROJ_ROOT = "D:\\bassl\\bassl"
    cfg.CKPT_PATH = os.path.join(cfg.PROJ_ROOT, "pretrain/ckpt")
    cfg.LOG_PATH = os.path.join(cfg.PROJ_ROOT, "pretrain/logs")
    if cfg.DATASET == "movienet":
        cfg.DATA_PATH = "D:\\bassl\\bassl\data\movienet"
        cfg.IMG_PATH = os.path.join(cfg.DATA_PATH, "240P_frames")
        cfg.ANNO_PATH = os.path.join(cfg.DATA_PATH, "anno")
        cfg.FEAT_PATH = os.path.join(cfg.DATA_PATH, "features")
    else:
        raise NotImplementedError

    # dry-run mode
    if cfg.DRY_RUN:
        cfg.TRAINER.limit_train_batches = 1 / 10000.0
        cfg.TRAINER.limit_val_batches = 1 / 1000.0
        cfg.TRAINER.num_sanity_val_steps = 0

    return cfg
