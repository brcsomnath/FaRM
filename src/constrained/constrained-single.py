#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

sys.path.append("../")
sys.path.append("src/")

import pickle
from copy import deepcopy
from random import *

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizerFast
from utils.loss import RateDistortionConstrained
from utils.mdl import MDL
from utils.net import Classifier
from utils.utils import *


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default="pan16",
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
                        default=4,
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
    parser.add_argument(
        "--model_save_path",
        default="saved_models/",
        type=str,
        help="Save path of the models.",
    )
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("--DATA_IDX", default=0, type=int, help="Data ID")
    parser.add_argument("--corrupt_labels", default=False, type=bool)
    parser.add_argument("--corruption_ratio", default=0.5, type=float)
    parser.add_argument("--truncate_train", default=False, type=bool)
    parser.add_argument("--truncation_ratio", default=1, type=float)
    parser.add_argument("--lambda_", default=0.01, type=float)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint_epoch", default=5, type=int)
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
    for (sent, label, bias) in tqdm(dataset):
        sent_emb = torch.tensor(
            tokenizer.encode(
                sent,
                max_length=max_len,
                pad_to_max_length=True,
                truncation=True,
                add_special_tokens=True,
            ))
        encoded_dataset.append((sent_emb, label, bias))
    return encoded_dataset


def get_dataset(dataset):
    """
    Input: the dataset name args.dataset

    Returns: train, dev, test
    train: list of training instances
    dev: list of development instances
    test: list of test instances
    """

    path = "data"

    if dataset == "dial":
        DIAL_PATH = f"{path}/dial/"
        Y_LABELS = {"Positive": 0, "Negative": 1}
        Z_LABELS = {"male": 0, "female": 1}

        train = load_content(load_dump(DIAL_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(DIAL_PATH + "test.pkl"))

    elif dataset == "dial-mention":
        DIAL_PATH = f"{path}/dial-mention/"
        Y_LABELS = {"Positive": 0, "Negative": 1}
        Z_LABELS = {"male": 0, "female": 1}

        train = load_content(load_dump(DIAL_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(DIAL_PATH + "test.pkl"))

    elif dataset == "pan16":
        PAN16_PATH = f"{path}/pan16/"
        Y_LABELS = {"Positive": 0, "Negative": 1}
        Z_LABELS = {"male": 0, "female": 1}

        train = load_content(load_dump(PAN16_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(PAN16_PATH + "test.pkl"))

    elif dataset == "pan16-age":
        PAN16_PATH = f"{path}/pan16-age/"
        Y_LABELS = {"Positive": 0, "Negative": 1}
        Z_LABELS = {"male": 0, "female": 1}

        train = load_content(load_dump(PAN16_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(PAN16_PATH + "test.pkl"))

    elif dataset == "bios":
        BIOS_PATH = f"{path}/bios/"
        Y_LABELS = {}
        for i in range(28):
            Y_LABELS[i] = i
        Z_LABELS = {"male": 0, "female": 1}

        train = load_content(load_dump(BIOS_PATH + "train.pkl"))
        dev = []
        test = load_content(load_dump(BIOS_PATH + "test.pkl"))

    return train, dev, test, Y_LABELS, Z_LABELS


def dump_data(path, filename, dataset):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename), "wb") as file:
        pickle.dump(dataset, file)


if __name__ == "__main__":
    args = get_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # Tokenizer initialize
    tokenizer = BertTokenizerFast.from_pretrained(args.MODEL)

    # Form dataset
    train, dev, test, Y_LABELS, Z_LABELS = get_dataset(args.dataset)

    if args.truncate_train:
        train_orig = deepcopy(train)
        train = train[:int(args.truncation_ratio * len(train))]

    train_dataset, dev_dataset, test_dataset = (
        encode(train, tokenizer, args.max_len),
        encode(dev, tokenizer, args.max_len),
        encode(test, tokenizer, args.max_len),
    )

    if args.corrupt_labels:
        print("Corrupting labels ...")
        train_dataset_orig = deepcopy(train_dataset)
        num_classes = len(Z_LABELS)
        n_train = len(train_dataset)
        n_rand = int(len(train_dataset) * args.corruption_ratio)

        randomize_indices = np.random.choice(range(n_train),
                                             size=n_rand,
                                             replace=False)

        for i, idx in enumerate(randomize_indices):
            label = np.random.choice(np.arange(num_classes))
            while label == train_dataset[idx][2]:
                label = np.random.choice(np.arange(num_classes))
            train_dataset[idx] = (train_dataset[idx][0], train_dataset[idx][1],
                                  label)

    train_dataloader, test_dataloader = load(train_dataset,
                                             args.batch_size), load(
                                                 test_dataset, args.batch_size)

    bert_model = initialize_models(args, device)

    net = Classifier(args, Y_LABELS)
    net.to(device)

    criterion = RateDistortionConstrained(gam1=args.gam1,
                                          gam2=args.gam2,
                                          eps=args.eps)
    disc_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW([{
        "params": bert_model.parameters()
    }, {
        "params": net.parameters()
    }],
                            lr=2e-5,
                            betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.1)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    ## Training
    itr = tqdm(range(args.epochs))

    for epoch in itr:
        total_task_loss = 0
        total_bias_loss = 0
        total_loss = 0

        for step, (data, label, bias) in enumerate(train_dataloader):
            real_data = data.to(device)

            optimizer.zero_grad()
            features = LN(bert_model(real_data)[1])
            output = net(features)

            bias_loss, _, _ = criterion(args, features, bias)
            task_loss = disc_criterion(output, label.to(device))

            loss = args.lambda_ * bias_loss + task_loss
            loss.backward()

            total_task_loss += task_loss.item()
            total_bias_loss += bias_loss.item()
            total_loss += loss.item()

            itr.set_description(f"Loss = {(total_loss / (step + 1)):.6f} \
                                Task Loss = {(total_task_loss / (step + 1)):.6f} \
                                Bias Loss = {(total_bias_loss / (step + 1)):.6f}"
                                )
            optimizer.step()

        if epoch > 0 and (epoch + 1) % args.checkpoint_epoch == 0:
            if not os.path.exists(
                    os.path.join(args.model_save_path, args.dataset)):
                os.makedirs(os.path.join(args.model_save_path, args.dataset))

            if args.truncate_train:
                torch.save(
                    bert_model,
                    f"{args.model_save_path}/{args.dataset}/bert-model-{args.dataset}-{(epoch + 1)}-{args.truncation_ratio}.pb",
                )
                torch.save(
                    net,
                    f"{args.model_save_path}/{args.dataset}/net-{args.dataset}-{(epoch + 1)}-{args.truncation_ratio}.pb",
                )
            elif args.corrupt_labels:
                torch.save(
                    bert_model,
                    f"{args.model_save_path}/{args.dataset}/bert-model-{args.dataset}-{(epoch + 1)}-corrupt-{args.corruption_ratio}.pb",
                )
                torch.save(
                    net,
                    f"{args.model_save_path}/{args.dataset}/net-{args.dataset}-{(epoch + 1)}-corrupt-{args.corruption_ratio}.pb",
                )
            else:
                torch.save(
                    bert_model,
                    f"{args.model_save_path}/{args.dataset}/bert-model-{args.dataset}-{(epoch + 1)}-{args.lambda_}.pb",
                )
                torch.save(
                    net,
                    f"{args.model_save_path}/{args.dataset}/net-{args.dataset}-{(epoch + 1)}-{args.lambda_}.pb",
                )

        scheduler.step()
    print("training complete.")

    if args.corrupt_labels:
        train_dataloader = load(train_dataset_orig, args.batch_size)

    if args.truncate_train:
        train_dataset = encode(train_orig, tokenizer, args.max_len)
        train_dataloader = load(train_dataset, args.batch_size)

    train_dataset = generate_purged_dataset_constrained(
        args, train_dataloader, bert_model)
    test_dataset = generate_purged_dataset_constrained(args, test_dataloader,
                                                       bert_model)

    data_train = [d[0] for d in train_dataset]
    task_label_train = [d[1] for d in train_dataset]
    bias_label_train = [d[2] for d in train_dataset]

    data_test = [d[0] for d in test_dataset]
    task_label_test = [d[1] for d in test_dataset]
    bias_label_test = [d[2] for d in test_dataset]

    F1, P, R = evaluate_mlp(data_train, task_label_train, data_test,
                            task_label_test)
    print(f"Task: F1 - {F1} P - {P} R - {R} ")

    F1, P, R = evaluate_mlp(data_train, bias_label_train, data_test,
                            bias_label_test)
    print(f"Bias: F1 - {F1} P - {P} R - {R} ")

    mdl = MDL(train_dataset)

    print(f"Bias MDL: {mdl.get_score(args, num_labels=2, label_id=2)}")
    print(
        f"Task MDL: {mdl.get_score(args, num_labels=3, label_id=1)}"
    )  # TODO: this won't work for BIOS dataset (for task MDL set num_labels=28)
