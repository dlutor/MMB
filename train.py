# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse

import random
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


from models import *
from utils import get_loader, AsymmetricLoss, AsymmetricLossOptimized, FocalLoss_MultiLabel
from utils import set_seed, save_checkpoint, load_checkpoint, create_logger, check_dirs
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from timm.scheduler import PlateauLRScheduler

from collections import defaultdict

def get_accuracy(y, y_pre):
    #	print('metric_acc:  ' + str(round(metrics.accuracy_score(y, y_pre),4)))
    samples = len(y)
    count = 0.0
    for i in range(samples):
        y_true = 0
        all_y = 0
        for j in range(len(y[i])):
            if y[i][j] > 0 and y_pre[i][j] > 0:
                y_true += 1
            if y[i][j] > 0 or y_pre[i][j] > 0:
                all_y += 1
        if all_y <= 0:
            all_y = 1

        count += float(y_true) / float(all_y)
    acc = float(count) / float(samples)
    acc = round(acc, 4)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def setup(args):
    n_en = args.n_en
    if "," in args.n_en:
        n_en = list(map(int, args.n_en.split(",")))
    else:
        n_en = int(n_en)
    kargs = {
        "num_encoder_layers": n_en,
        "num_decoder_layers": args.n_de,
        "d_model": args.d_model,
        "nhead": args.n_head,
        "dim_ff_ratio": args.dim_ff_ratio,
    }
    if args.class_nums:
        kargs["num_cls"] = int(args.class_nums)
    if args.data_dims:
        data_dims = list(map(int, args.data_dims.split(",")))
        kargs["dims"] = data_dims
    if args.embed_dims:
        embed_dims = list(map(int, args.embed_dims.split(",")))
        kargs["encoder_dims"] = embed_dims
    if args.t_length:
        t_length = list(map(int, args.t_length.split(",")))
        kargs["t_length"] = t_length
    if args.t_mask == "y":
        kargs["mask"] = args.t_mask
    if args.encoder_heads:
        encoder_heads = list(map(int, args.encoder_heads.split(",")))
        kargs["encoder_heads"] = encoder_heads
    if args.fusion:
        kargs["fusion"] = eval(args.fusion)

    model = eval(args.model)(**kargs)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000




def evaluate(args, model, data_loader, is_save=False):

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    save_data = defaultdict(list)

    if args.verbose == "y":
        epoch_iterator = tqdm(data_loader,
                              desc="Validating... ",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
    else:
        epoch_iterator = data_loader

    for step, data in enumerate(epoch_iterator):
        batch_data = []
        for da in data:
            batch_data.append(da.float().to(args.device))
        labels = batch_data[-1]
        with torch.no_grad():
            logits = model(*batch_data[:-1])

        if isinstance(logits, tuple):
            logits, _ = logits
            for key in _.keys():
                data = _[key]
                if isinstance(data, list):
                    data = torch.stack(data).transpose(0, 1)
                data = data.cpu()
                save_data[key].append(data)
        preds = (logits > args.threshold).int().cpu().tolist()
        logits = logits.cpu().tolist()

        all_preds += preds
        all_labels += labels.int().tolist()
        all_logits += logits

    for key in save_data.keys():
        save_data[key] = torch.vstack(save_data[key])
    if is_save:
        torch.save(save_data, f"{args.path}/save.pt")

    accuracy = get_accuracy(all_labels, all_preds)
    hl = hamming_loss(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='micro')
    r = recall_score(all_labels, all_preds, average='micro')
    mif1 = f1_score(all_labels, all_preds, average='micro')
    maf1 = f1_score(all_labels, all_preds, average='macro')
    p1 = precision_score((1 - torch.tensor(all_labels, dtype=torch.int)).tolist(), (1 - torch.tensor(all_preds, dtype=torch.int)).tolist(), average='micro')
    r1 = recall_score((1 - torch.tensor(all_labels, dtype=torch.int)).tolist(), (1 - torch.tensor(all_preds, dtype=torch.int)).tolist(), average='micro')
    results = {
        "accuracy":accuracy,
        "hl":hl,
        "p":p,
        "r":r,
        "mif1":mif1,
        "maf1":maf1,
        "p1": p1,
        "r1": r1,
    }
    logger.info(fomart_results(results))
    return results



def fomart_results(results):
    str_ = [
        f"Accuracy: {results['accuracy']*100:.2f}", f"HammingLoss: {results['hl']:.4f}",
        f"Precision: {results['p']*100:.2f}", f"Recall: {results['r']*100:.2f}",
        f"Micro F1: {results['mif1']*100:.2f}", f"Macro F1: {results['maf1']*100:.2f}",
        f"Precision1: {results['p1']*100:.2f}", f"Recall1: {results['r1']*100:.2f}",
            ]
    str_ = "\n".join(str_)
    return str_

def sfomart_results(results):
    strs = ["Acc.", "HLos", "Prec", "Rec.", "MiF1", "MaF1", "Prec1", "Rec.1"]
    str_ = [
        f"{strs[0]}: {results['accuracy']*100:.2f}", f"{strs[1]}: {results['hl']:.4f}",
        f"{strs[2]}: {results['p']*100:.2f}", f"{strs[3]}: {results['r']*100:.2f}",
        f"{strs[4]}: {results['mif1']*100:.2f}", f"{strs[5]}: {results['maf1']*100:.2f}",
        f"{strs[6]}: {results['p1']*100:.2f}", f"{strs[7]}: {results['r1']*100:.2f}",
    ]
    str_ = "\n".join(str_)
    return str_

def train(args, model, train_loader, dev_loader, test_loader):
    """ Train the model """

    if args.optimizer == "AdamW":
        optim = torch.optim.AdamW
    else:
        optim = torch.optim.Adam
    optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)


    scheduler = PlateauLRScheduler(optimizer, decay_rate=args.lr_factor, patience_t=args.lr_patience,
                                   warmup_t=args.warmup, warmup_lr_init=5e-6)

    if args.loss == "ASL":
        loss_fn = AsymmetricLossOptimized(gamma_neg=args.ASL_neg_gamma, gamma_pos=args.ASL_pos_gamma, clip=args.ASL_clip)
    elif args.loss == "BCE":
        loss_fn = nn.BCELoss()
    else:
        loss_fn = eval(args.loss)
    loss_fn_kargs = {}
    if args.loss == "lq_loss":
        loss_fn_kargs["q_pos_h"] = args.lq_pos_h
        loss_fn_kargs["q_pos_e"] = args.lq_pos_e
        loss_fn_kargs["q_neg_h"] = args.lq_neg_h
        loss_fn_kargs["gamma_pos"] = args.lq_pos_gamma
        loss_fn_kargs["gamma_neg"] = args.lq_neg_gamma

    wloss_kargs = {}
    if args.wloss:
        wloss_kargs["alpha_pos"] = args.wloss_alpha_pos
        wloss_kargs["beta_pos"] = args.wloss_beta_pos
        wloss_kargs["alpha_neg"] = args.wloss_alpha_neg
        wloss_kargs["beta_neg"] = args.wloss_beta_neg


    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size / args.n_gpu)

    model.zero_grad()

    best_dev_f1, start_epoch, best_epoch = 0, 0, 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.path, "checkpoint.pt")):
        logger.info(f"load checkpoint {os.path.join(args.path, 'checkpoint.pt')}")
        checkpoint = torch.load(os.path.join(args.path, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_dev_f1 = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])


    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if args.verbose == "y":
            epoch_iterator = tqdm(train_loader,
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  disable=args.local_rank not in [-1, 0])
        else:
            epoch_iterator = train_loader
        losses = AverageMeter()
        global_step = 0
        optimizer.zero_grad()
        for step, data in enumerate(epoch_iterator):#

            batch_data = []
            for da in data:
                batch_data.append(da.float().to(args.device))
            labels = batch_data[-1]
            preds = model(*batch_data[:-1])

            r_dict = None
            if isinstance(preds, tuple):
                preds, r_dict = preds

            head = r_dict["head"]


            if args.wloss != 0 and r_dict is not None:
                head = WLoss(r_dict, labels, **wloss_kargs)
                preds = torch.sigmoid(head)
            if args.loss in ["ASL", "BCE"]:
                loss = loss_fn(preds, labels, **loss_fn_kargs)
            else:
                loss = loss_fn(head, labels, **loss_fn_kargs)

            if args.wfeats != 0  and r_dict is not None:
                preds, return_dict = wfeats(r_dict["feats_"], sum(r_dict["feats"]), model)
                head = return_dict["head"]
                if args.loss in ["ASL", "BCE"]:
                    loss += loss_fn(preds, labels, **loss_fn_kargs) * args.wfeats
                else:
                    loss += loss_fn(head, labels, **loss_fn_kargs) * args.wfeats


            if r_dict is not None:
                if isinstance(r_dict, dict):
                    preds_ = r_dict["logits_"]
                else:
                    preds_ = r_dict
                if args.alpha != 0:
                    for pred_ in preds_:
                        loss += loss_fn(pred_, labels, **loss_fn_kargs) * args.alpha
                if args.d_reg != 0:
                    if args.d_reg_type == "feats":
                        loss += d_reg(r_dict["feats_"], sum(r_dict["feats"]), d_fun=eval(args.d_reg_fun)) * args.d_reg
                    else:
                        loss += d_reg(r_dict["heads_"], r_dict["head"], d_fun=eval(args.d_reg_fun)) * args.d_reg

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses.update(loss.item() * args.gradient_accumulation_steps)
            if args.verbose == "y":
                epoch_iterator.set_description(
                    "Training (%d / %d Epochs)(cur_loss=%2.4f, avg_loss=%2.4f)" % (
                        epoch + 1, args.num_epochs, losses.val, losses.avg))

        # scheduler.step(epoch)
        if global_step % args.gradient_accumulation_steps != 0:
            logger.info(f'Step drop batch {global_step % args.gradient_accumulation_steps}')
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f'[{epoch + 1:2d}/{args.num_epochs}] Evaluating on dev set......')
        dev_results = evaluate(args, model, dev_loader)
        dev_f1 = dev_results["accuracy"]

        scheduler.step(epoch, metric=dev_f1)

        is_improvement = dev_f1 > best_dev_f1
        if is_improvement:
            # save_model(args, model)
            best_dev_f1 = dev_f1
            best_epoch = epoch + 1
            logger.info("Saved best model checkpoint to [DIR: %s]", args.path)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_dev_f1,
                "best_epoch": best_epoch,
            },
            is_improvement,
            args.path,
        )
        if epoch + 1 - best_epoch >= args.patience:
            logger.info(f"After {args.patience} epochs not improve, break training.")
            break

    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="MOSEI_Aligned",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["Aligned",  "NEMu", "UnAligned"],
                        default="Aligned",
                        help="Which downstream task.")
    parser.add_argument("--com_max_length", default=1024, type=int)
    parser.add_argument("--lyr_max_length", default=512, type=int)

    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight delay if we apply some.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Num of training epochs.")
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Whether to train or validate the model.')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='the threshold of whether the emotion exists.')
    parser.add_argument('--n_en', default="4", type=str,
                        help='Num of uni- or multi-modal encoders.')
    parser.add_argument('--n_de', default=2, type=int,
                        help='Num of decoders.')
    parser.add_argument('--d_model', default=224, type=int,
                        help=' the number of expected features in the transformer input.')
    parser.add_argument('--n_head', default=8, type=int,
                        help=' the number of head in the transformer.')
    parser.add_argument('--dim_ff_ratio', default=2, type=float,
                        help=' the dimension of the feed forward network model.')

    parser.add_argument("--model", default="MLPModel", type=str,
                        help="The model.")
    parser.add_argument("--class_nums", default="", type=str,
                        help="The data class nums.")
    parser.add_argument("--t_length", default="", type=str,
                        help="The data time length.")
    parser.add_argument("--t_mask", default="", type=str,
                        help="mask zero data in time dims.")
    parser.add_argument("--data_dims", default="", type=str,
                        help="The data dims.")
    parser.add_argument("--embed_dims", default="", type=str,
                        help="The embedding dims.")
    parser.add_argument("--encoder_heads", default="", type=str,
                        help="The embedding dims.")
    parser.add_argument("--loss", default="ASL", type=str,
                        help="The loss function.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument('--num_workers', default=0, type=int,
                        help='dataloader num_workers.')
    parser.add_argument('--alpha', default=0., type=float,
                        help='the alpha of unimodel loss.')
    parser.add_argument('--load_checkpoint', default=0, type=int,
                        help='load checkpoint on training.')

    parser.add_argument('--wloss', default=1, type=float,
                        help='wloss.')

    parser.add_argument('--wfeats', default=0, type=float,
                        help='wfeats.')

    parser.add_argument('--d_reg', default=0, type=float,
                        help='d_reg loss.')
    parser.add_argument('--d_reg_type', default="feats", type=str)
    parser.add_argument('--d_reg_fun', default="cos_d", type=str)


    parser.add_argument("--fusion", default="sum", type=str,
                        help="The fusion method.")
    parser.add_argument("--gpu", default="0", type=str,
                        help="The gpu used.")
    parser.add_argument("--optimizer", default="AdamW", type=str,
                        help="The optimizer used.")

    parser.add_argument('--warmup', default=10, type=int,
                        help='warmup.')
    parser.add_argument('--patience', default=10, type=int,
                        help='patience.')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='lr_patience.')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='lr_factor .')

    parser.add_argument('--lq_pos_e', default=0.1, type=float)
    parser.add_argument('--lq_pos_h', default=0.01, type=float)
    parser.add_argument('--lq_neg_e', default=1, type=float)
    parser.add_argument('--lq_neg_h', default=1, type=float)
    parser.add_argument('--lq_pos_gamma', default=1, type=float)
    parser.add_argument('--lq_neg_gamma', default=1, type=float)

    parser.add_argument('--ASL_pos_gamma', default=1, type=float)
    parser.add_argument('--ASL_neg_gamma', default=2, type=float)
    parser.add_argument('--ASL_clip', default=0.05, type=float)


    parser.add_argument('--wloss_alpha_pos', default=3, type=float)
    parser.add_argument('--wloss_beta_pos', default=2.5, type=float)
    parser.add_argument('--wloss_alpha_neg', default=2, type=float)
    parser.add_argument('--wloss_beta_neg', default=2, type=float)

    parser.add_argument("--verbose", default="n", type=str,
                        help="mask zero data in time dims.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(seconds=30))
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    args.path = f"./{args.output_dir}/{args.name}"
    check_dirs(args.path)
    global logger
    logger = create_logger(f"{args.path}/logfile.log", args)

    set_seed(args.seed)
    args, model = setup(args)
    train_loader, dev_loader, test_loader = get_loader(args)

    if not args.eval_only:
        time_start = time.time()
        train(args, model, train_loader, dev_loader, test_loader)
        time_end = time.time()
        logger.info('Training time cost: %2.1f minutes.' % ((time_end - time_start) / 60))

    # Validating
    # model = load_model(args, model)

    logger.info(f"Evaluating on test set......")
    checkpoint = load_checkpoint(model, f"{args.path}/model_best.pt")
    logger.info(f"load best checkpoint epoch: {checkpoint['epoch']}")
    # evaluate(args, model, dev_loader)
    results = evaluate(args, model, test_loader, is_save=True)
    logger.info(sfomart_results(results))
    print(sfomart_results(results))



if __name__ == "__main__":
    main()
