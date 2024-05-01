import os
import sys
import peft
import hydra
import logging
import importlib
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any
from omegaconf import DictConfig
from torch.func import jacrev, functional_call
from torchmetrics import Accuracy, CalibrationError
from transformers.modeling_outputs import ModelOutput

from utils import dsets
from utils.loggers import setup_loggers
from utils.setup_llm import setup_llm


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="example_usage",
)
def main(cfg: DictConfig):
    #
    # 1. Load configuration from Hydra
    #
    device = "cuda:0"
    setup_loggers(cfg)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    #
    # 2. Load PEFT model and dataset
    #
    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    # model = model.to(device)
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dset = dset_class(tokenizer, add_space=cfg.llm.add_space)

    #
    # 3. Do MAP training
    #
    train_loader = dset.loader(
        is_s2s=cfg.llm.is_s2s,  # sequence to sequence model?
        batch_size=cfg.dset.train_bs,  # training batch size
        split=cfg.dset.train_split,  # training split name in dset
        subset_size=cfg.dset.train_subset,  # train on subset? (-1 = no subset)
    )
    map_param_path = f"{cfg.paths.output_dir}/MAP_params.pth"
    grad_steps, epoch = 0, 0
    if not os.path.exists(map_param_path) or cfg.run_every_step:
        # setup optimiser
        opt_cfg = dict(cfg.opt)
        # add prior / regularization for MAP objective:
        opt_cfg.update({"weight_decay": 1 / cfg.prior_var})
        optclass = getattr(
            importlib.import_module(opt_cfg.pop("module")),
            opt_cfg.pop("classname"),
        )
        opt = optclass(model.parameters(), **opt_cfg)
        logging.info("Training MAP parameters")
        while grad_steps < cfg.train_steps:
            epoch += 1
            logging.info(f"Beginning epoch {epoch} ({grad_steps} / {cfg.train_steps})")
            for batch in tqdm(train_loader, disable=not cfg.use_tqdm, file=sys.stdout):
                opt.zero_grad()
                prompts, classes, _ = batch
                inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)
                logits = model(**inputs).logits[:, -1, dset.target_ids.squeeze(-1)]
                # loss = F.cross_entropy(logits[:, -1], targets.to(device))
                loss = F.cross_entropy(logits, classes.to(device))
                assert not t.isnan(loss).any(), "NaN in loss for MAP training."
                loss.backward()
                opt.step()
                grad_steps += 1
                if not grad_steps < cfg.train_steps:
                    break
        logging.info(f"Saving MAP parameters after finetuning to {map_param_path}")
        model.save_pretrained(map_param_path)
    else:
        logging.info(f"Loading MAP parameters from {map_param_path}")
        del model
        llm_params = dict(cfg.llm) | {"use_peft": False}
        model, _, _ = setup_llm(**llm_params)
        model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
        model = model.to(device)

    val_loader = dset.loader(
        is_s2s=cfg.llm.is_s2s,
        batch_size=cfg.dset.eval_bs,
        split=cfg.dset.eval_split,
        subset_size=cfg.dset.eval_subset,
    )


    total_loss = 0
    metric_kwargs = {"task": "multiclass", "num_classes": dset.n_labels}
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)
    print(model)

    with t.no_grad():
        for batch in tqdm(val_loader, disable=not cfg.use_tqdm, file=sys.stdout):
            prompts, classes, _ = batch
            classes = classes.to(device)

            batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)

            logits = model(**batch_inputs).logits[:, -1, dset.target_ids.squeeze(-1)]
            total_loss += F.cross_entropy(logits, classes).item()
            acc_metric(logits, classes)
            ece_metric(logits, classes)

    loss = total_loss / len(val_loader)
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()
    print(loss, acc, ece)
    logging.info(f"NLL: {loss:.5f}, ACC: {acc:.5f}, ECE: {ece:.5f}")

    logging.info("Successfully finished.")


if __name__ == "__main__":
    main()