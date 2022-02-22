#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.append("../")
sys.path.append("src/")
from random import *

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizerFast
from utils.loss import RateDistortionConstrainedMultiple
from utils.mdl import MDL
from utils.net import Classifier
from utils.utils import *

sys.path.append("../")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default="pan16-dual",
                        type=str,
                        help="Dataset name.")
    parser.add_argument(
        "--MODEL",
        default="bert-base-uncased",
        type=str,
        help="Name of the BERT model to be used as the text encoder.",
    )
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="Batch Size.")
    parser.add_argument("--num_layers",
                        default=2,
                        type=int,
                        help="Number of layers.")
    parser.add_argument(
        "--max_len",
        default=32,
        type=int,
        help="Maximum length of the sequence after tokenization.",
    )
    parser.add_argument(
        "--embedding_size",
        default=768,
        type=int,
        help="Hidden size of the representation output of the text encoder.",
    )
    parser.add_argument("--device",
                        default="cuda:3",
                        type=str,
                        help="GPU device.")
    parser.add_argument("--partition",
                        default="one",
                        type=str,
                        help="Type of partition.")
    parser.add_argument(
        "--model_save_path",
        default="saved_models/multiple/",
        type=str,
        help="Save path of the models.",
    )
    parser.add_argument("--epochs",
                        default=25,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("--DATA_IDX", default=0, type=int, help="Data ID")
    parser.add_argument("--Y_IDX", default=1, type=int, help="Y ID")
    parser.add_argument("--Z1_IDX", default=2, type=int, help="Z1 ID")
    parser.add_argument("--Z2_IDX", default=3, type=int, help="Z2 ID")
    parser.add_argument("--checkpoint",
                        default=5,
                        type=int,
                        help="Save after these epochs.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--mom",
                        type=float,
                        default=0.9,
                        help="momentum (default: 0.9)")
    parser.add_argument("--wd",
                        type=float,
                        default=5e-4,
                        help="weight decay (default: 5e-4)")
    parser.add_argument("--lamdba",
                        type=float,
                        default=0.01,
                        help="Hyperparameter for bias loss")
    parser.add_argument(
        "--gam1",
        type=float,
        default=1.0,
        help="gamma1 for tuning empirical loss (default: 1.)",
    )
    parser.add_argument(
        "--gam2",
        type=float,
        default=1.0,
        help="gamma2 for tuning empirical loss (default: 1.)",
    )
    parser.add_argument("--eps",
                        type=float,
                        default=0.5,
                        help="eps squared (default: 0.5)")
    return parser


def encode(dataset, tokenizer, max_len):
    """Converts data into token representations in the form of tensors."""
    encoded_dataset = []
    for (sent, label, bias_1, bias_2) in tqdm(dataset):
        sent_emb = torch.tensor(
            tokenizer.encode(
                sent,
                max_length=max_len,
                pad_to_max_length=True,
                truncation=True,
                add_special_tokens=True,
            ))
        encoded_dataset.append((sent_emb, label, bias_1, bias_2))
    return encoded_dataset


def load_content(content, DATA_IDX=1, Y_IDX=2, Z1_IDX=3, Z2_IDX=4):
    """
    Forms the data in the format (sentence, y-label, z-label)
    Returns an array of instances in the above format
    """
    dataset = []
    for c in content:
        dataset.append((c[DATA_IDX], c[Y_IDX], c[Z1_IDX], c[Z2_IDX]))
    return dataset


def get_dataset(dataset):
    """
    Input: the dataset name args.dataset

    Returns: train, dev, test
    train: list of training instances
    dev: list of development instances
    test: list of test instances
    """

    if dataset == "pan16-dual":
        PAN16_PATH = "../../purging-embeddings/data/pan16/dual/"

        Y_LABELS = {"Positive": 0, "Negative": 1}
        Z1_LABELS = {"male": 0, "female": 1}
        Z2_LABELS = {"young": 0, "old": 1}

        train = load_content(load_dump(PAN16_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(PAN16_PATH + "test.pkl"))

    return train, dev, test, Y_LABELS, Z1_LABELS, Z2_LABELS


if __name__ == "__main__":

    args = get_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # Tokenizer initialize
    tokenizer = BertTokenizerFast.from_pretrained(args.MODEL)

    # Form dataset
    train, dev, test, Y_LABELS, Z1_LABELS, Z2_LABELS = get_dataset(
        args.dataset)

    train_dataset, dev_dataset, test_dataset = (
        encode(train, tokenizer, args.max_len),
        encode(dev, tokenizer, args.max_len),
        encode(test, tokenizer, args.max_len),
    )
    train_dataloader, test_dataloader = load(train_dataset,
                                             args.batch_size), load(
                                                 test_dataset, args.batch_size)

    # Model
    bert_model = initialize_models(args, device)

    net = Classifier(args, Y_LABELS)
    net.to(device)

    criterion = RateDistortionConstrainedMultiple(gam1=args.gam1,
                                                  gam2=args.gam2,
                                                  eps=args.eps)
    disc_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW([{
        "params": bert_model.parameters()
    }],
                            lr=2e-5,
                            betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.1)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    # Training
    itr = tqdm(range(args.epochs))

    epoch = 0
    for epoch in itr:
        total_task_loss = 0
        total_bias_loss = 0
        total_loss = 0

        for step, (data, label, bias_1, bias_2) in enumerate(train_dataloader):
            real_data = data.to(device)

            optimizer.zero_grad()
            features = LN(bert_model(real_data)[1])
            output = net(features)

            bias_loss = criterion(args, features, bias_1, bias_2)
            task_loss = disc_criterion(output, label.to(device))

            loss = args.lamdba * bias_loss + task_loss
            loss.backward()

            total_task_loss += task_loss.item()
            total_bias_loss += bias_loss.item()
            total_loss += loss.item()

            itr.set_description(f"Loss = {(total_loss / (step + 1)):.6f} \
                                Task Loss = {(total_task_loss / (step + 1)):.6f} \
                                Bias Loss = {(total_bias_loss / (step + 1)):.6f}"
                                )
            optimizer.step()

        if epoch > 0 and (epoch + 1) % args.checkpoint == 0:
            if not os.path.exists(
                    os.path.join(args.model_save_path, args.dataset)):
                os.makedirs(os.path.join(args.model_save_path, args.dataset))

            torch.save(
                bert_model,
                f"{args.model_save_path}/{args.dataset}/bert-model-int-{args.dataset}-{(epoch + 1)}.pb",
            )
            torch.save(
                net,
                f"{args.model_save_path}/{args.dataset}/net-int-{args.dataset}-{(epoch + 1)}.pb",
            )

        scheduler.step()
    print("training complete.")

    if not os.path.exists(os.path.join(args.model_save_path, args.dataset)):
        os.makedirs(os.path.join(args.model_save_path, args.dataset))

    torch.save(
        bert_model,
        f"{args.model_save_path}/{args.dataset}/bert-model-{args.partition}-{args.dataset}-{(epoch + 1)}.pb",
    )
    torch.save(
        net,
        f"{args.model_save_path}/{args.dataset}/net-{args.partition}-{args.dataset}-{(epoch + 1)}.pb",
    )

    train_dataset = generate_purged_dataset_multiple(args, train_dataloader,
                                                     bert_model)
    test_dataset = generate_purged_dataset_multiple(args, test_dataloader,
                                                    bert_model)

    data_train = [d[args.DATA_IDX] for d in train_dataset]
    task_label_train = [d[args.Y_IDX] for d in train_dataset]
    bias1_label_train = [d[args.Z1_IDX] for d in train_dataset]
    bias2_label_train = [d[args.Z2_IDX] for d in train_dataset]

    data_test = [d[args.DATA_IDX] for d in test_dataset]
    task_label_test = [d[args.Y_IDX] for d in test_dataset]
    bias1_label_test = [d[args.Z1_IDX] for d in test_dataset]
    bias2_label_test = [d[args.Z2_IDX] for d in test_dataset]

    # Task Performance
    F1, P, R = evaluate_mlp(data_train, task_label_train, data_test,
                            task_label_test)
    print(f"Task: F1 - {F1} P - {P} R - {R} ")

    # Bias-1 Accuracy:

    F1, P, R = evaluate_mlp(data_train, bias1_label_train, data_test,
                            bias1_label_test)
    print(f"Bias_1: F1 - {F1} P - {P} R - {R} ")

    # Bias-2 Accuracy:
    F1, P, R = evaluate_mlp(data_train, bias2_label_train, data_test,
                            bias2_label_test)
    print(f"Bias_2: F1 - {F1} P - {P} R - {R} ")

    mdl = MDL(train_dataset)

    print(f"Bias - 1 MDL: {mdl.get_score(args, num_labels=2, label_id=2)}")
    print(f"Bias - 2 MDL: {mdl.get_score(args, num_labels=2, label_id=3)}")
    print(f"Task MDL: {mdl.get_score(num_labels=2, label_id=1)}")
