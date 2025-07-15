# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    apex_available = True
except ImportError:
    apex_available = False
    print("[WARN] Apex not found. FP16 training will be disabled.")

from models.modeling import CmdVIT, CONFIGS
from models.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from data_utils import get_loader
from dist_util import get_world_size

from sklearn import metrics

logger = logging.getLogger(__name__)

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()

    def forward(self, points, centers):
        points_expanded = points.unsqueeze(2)
        centers_expanded = centers.unsqueeze(1)
        distances = torch.norm(points_expanded - centers_expanded, p=2, dim=3)
        min_distances, _ = torch.min(distances, dim=2)
        return torch.mean(min_distances)


class AverageMeter:
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    os.makedirs(args.output_dir, exist_ok=True)
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved model checkpoint to [DIR: {args.output_dir}]")


def setup(args):
    config = CONFIGS[args.model_type]
    num_classes = 7

    model = CmdVIT(config, args.img_size, zero_head=True, num_classes=num_classes)
    logger.info(f"Loading pretrained weights from {args.pretrained_dir}")
    if args.pretrained_dir:
        logger.warning(f"Ignoring pretrained weights because model doesn't support .npz loading.")
    model.to(args.device)

    num_params = count_parameters(model)
    logger.info(f"Model Config:\n{config}")
    logger.info(f"Total Parameters: {num_params:.1f}M")
    return args, model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    model.eval()

    logger.info("***** Running Validation *****")
    logger.info(f"  Num steps = {len(test_loader)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    all_preds, all_labels = [], []
    loss_fct = nn.CrossEntropyLoss(weight=args.class_weights.to(args.device))

    epoch_iterator = tqdm(test_loader, desc="Validating... (loss=X.X)", dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            outputs = model(x)
            logits = outputs[0]
            atten = outputs[2] if len(outputs) > 2 else None

            eval_loss = loss_fct(logits, y)
            if atten is not None:
                cl_loss = CenterLoss().forward(atten[0], atten[1]).to(args.device)
                eval_loss = eval_loss + 0.1 * cl_loss

        eval_losses.update(eval_loss.item())
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = simple_accuracy(all_preds, all_labels)
    with open('class.txt', encoding='utf-8') as f:
        target_names = [x.strip() for x in f.readlines()]
    report = metrics.classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
    logger.info(report)

    logger.info(f"Validation Results - Global Steps: {global_step}")
    logger.info(f"Valid Loss: {eval_losses.avg:.5f}")
    logger.info(f"Valid Accuracy: {accuracy:.5f}")

    if writer:
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    else:
        writer = None

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    train_loader, test_loader, class_weights = get_loader(args)
    args.class_weights = class_weights

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    t_total = args.num_steps
    scheduler_cls = WarmupCosineSchedule if args.decay_type == "cosine" else WarmupLinearSchedule
    scheduler = scheduler_cls(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16 and apex_available:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        logger.info("Using Apex AMP for FP16 training.")
    elif args.fp16:
        logger.warning("FP16 was requested but Apex is not installed. Continuing in FP32.")

    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    logger.info("***** Running training *****")
    logger.info(f"  Total steps = {args.num_steps}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Accumulation steps = {args.gradient_accumulation_steps}")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0

    while global_step < t_total:
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            outputs = model(x, y)
            loss = outputs[0]
            atten = outputs[1] if len(outputs) > 1 else None

            if atten is not None:
                cl_loss = CenterLoss().forward(atten[0], atten[1]).to(args.device)
                loss = loss + 0.1 * cl_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16 and apex_available:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    f"Training ({global_step} / {t_total} Steps) (loss={losses.val:.5f})"
                )
                if writer and args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step >= t_total:
                    break
        losses.reset()

    if writer and args.local_rank in [-1, 0]:
        writer.close()
    logger.info(f"Best Accuracy: {best_acc:.5f}")
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="CmdVITRun", help="Run name.")
    parser.add_argument("--model_type", default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--eval_every", default=30, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", default=150, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O2')
    parser.add_argument('--loss_scale', type=float, default=0)

    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, "
                   f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
