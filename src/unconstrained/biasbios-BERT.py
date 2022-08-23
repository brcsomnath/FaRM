#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import os
import pickle
import warnings

sys.path.append("../")
sys.path.append("src/")

from collections import Counter, defaultdict
from copy import deepcopy
from random import *
from typing import List
from utils.utils import *
from utils.loss import RateDistortionUnconstrained
from utils.mdl import MDL
from utils.net import Net

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch import nn, optim
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size",
                        type=int,
                        default=768,
                        help="eembedding_size ")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--dataset", default="bios-bert", type=str)
    parser.add_argument(
        "--data_path",
        default="data/biasbios",
        type=str,
    )
    parser.add_argument(
        "--encoded_data_path",
        default="data/bert_encode_biasbios",
        type=str,
    )
    parser.add_argument("--model_save_path",
                        default="saved_models/unconstrained/",
                        type=str)
    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        help="Batch Size.")
    parser.add_argument("--num_layers",
                        default=4,
                        type=int,
                        help="Network layers.")
    parser.add_argument("--corrupt_labels", default=False, type=bool)
    parser.add_argument("--corruption_ratio", default=0.5, type=float)
    parser.add_argument("--truncate_train", default=False, type=bool)
    parser.add_argument("--truncation_ratio", default=1, type=float)
    parser.add_argument("--DATA_IDX", default=0, type=int, help="Data ID")
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


def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_dictionary(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    k2v, v2k = {}, {}
    for line in lines:

        k, v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k

    return k2v, v2k


def count_profs_and_gender(data: List[dict]):

    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1

    return counter


if __name__ == "__main__":
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading data ...")
    train = load_dataset(f"{args.data_path}/train.pickle")
    dev = load_dataset(f"{args.data_path}/dev.pickle")
    test = load_dataset(f"{args.data_path}/test.pickle")
    counter = count_profs_and_gender(train + dev + test)
    p2i, i2p = load_dictionary(f"{args.data_path}/profession2index.txt")
    g2i, i2g = load_dictionary(f"{args.data_path}/gender2index.txt")

    x_train = np.load(f"{args.encoded_data_path}/train_cls.npy")
    x_dev = np.load(f"{args.encoded_data_path}/dev_cls.npy")
    x_test = np.load(f"{args.encoded_data_path}/test_cls.npy")

    assert len(train) == len(x_train)
    assert len(dev) == len(x_dev)
    assert len(test) == len(x_test)

    f, m = 0.0, 0.0
    prof2fem = dict()

    for k, values in counter.items():
        f += values["f"]
        m += values["m"]
        prof2fem[k] = values["f"] / (values["f"] + values["m"])

    print(f / (f + m))
    print(prof2fem)

    y_train = np.array([p2i[entry["p"]] for entry in train])
    y_dev = np.array([p2i[entry["p"]] for entry in dev])
    y_test = np.array([p2i[entry["p"]] for entry in test])

    Y_dev_gender = np.array([g2i[d["g"]] for d in dev])
    Y_test_gender = np.array([g2i[d["g"]] for d in test])
    Y_train_gender = np.array([g2i[d["g"]] for d in train])

    print("Data loaded!")

    net = Net(args)
    net.to(device)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    if args.truncate_train:
        x_train_orig, y_train_orig, Y_train_gender_orig = (
            deepcopy(x_train),
            deepcopy(y_train),
            deepcopy(Y_train_gender),
        )
        x_train = x_train[:int(args.truncation_ratio * len(x_train))]
        y_train = y_train[:int(args.truncation_ratio * len(y_train))]
        Y_train_gender = Y_train_gender[:int(args.truncation_ratio *
                                             len(Y_train_gender))]

    train_dataset = encode(x_train, y_train, Y_train_gender)
    dev_dataset = encode(x_dev, y_dev, Y_dev_gender)
    test_dataset = encode(x_test, y_test, Y_test_gender)

    if args.corrupt_labels:
        print("Corrupting labels ...")
        train_dataset_orig = deepcopy(train_dataset)
        num_classes = max(Y_train_gender) + 1
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
        print("Done!")

    train_dataloader, dev_dataloader, test_dataloader = (
        load(train_dataset, args.batch_size),
        load(dev_dataset, args.batch_size),
        load(test_dataset, args.batch_size),
    )

    criterion = RateDistortionUnconstrained(gam1=args.gam1,
                                            gam2=args.gam2,
                                            eps=args.eps)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.mom,
                          weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400, 600], gamma=0.1)

    print("Starting training ...")
    itr = tqdm(range(args.epochs))

    for _, epoch in enumerate(itr, 0):
        total_loss = 0
        for step, (batch_embs, batch_y,
                   batch_z) in enumerate(train_dataloader):
            features = LN(net(batch_embs.to(device)))
            loss = criterion(args, features, batch_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss -= loss.item()
            itr.set_description(
                f"Epoch = {epoch} Loss = {(total_loss / (step + 1)):.6f}")

        scheduler.step()

    print("Training complete!")

    if not os.path.exists(os.path.join(args.model_save_path, args.dataset)):
        os.makedirs(os.path.join(args.model_save_path, args.dataset))

    if args.corrupt_labels:
        torch.save(
            net,
            f"{args.model_save_path}/{args.dataset}/net-{args.dataset}-corrupt-{args.corruption_ratio}.pb",
        )
    elif args.truncate_train:
        torch.save(
            net,
            f"{args.model_save_path}/{args.dataset}/net-{args.dataset}-truncate-{args.truncation_ratio}.pb",
        )
    else:
        torch.save(
            net,
            f"{args.model_save_path}/{args.dataset}/net-{args.dataset}.pb")

    if args.corrupt_labels:
        train_dataloader = load(train_dataset_orig, args.batch_size)

    if args.truncate_train:
        x_train, y_train, Y_train_gender = (
            deepcopy(x_train_orig),
            deepcopy(y_train_orig),
            deepcopy(Y_train_gender_orig),
        )

        train_dataset = encode(x_train, y_train, Y_train_gender)
        dev_dataset = encode(x_dev, y_dev, Y_dev_gender)
        test_dataset = encode(x_test, y_test, Y_test_gender)

        train_dataloader, dev_dataloader, test_dataloader = (
            load(train_dataset, args.batch_size),
            load(dev_dataset, args.batch_size),
            load(test_dataset, args.batch_size),
        )

    train_dataset = generate_debiased_embeddings(args, train_dataloader, net)
    dev_dataset = generate_debiased_embeddings(args, dev_dataloader, net)
    test_dataset = generate_debiased_embeddings(args, test_dataloader, net)

    data_train = np.array([d[0] for d in train_dataset])
    Y_train = np.array([d[1] for d in train_dataset])
    Z_train = np.array([d[2] for d in train_dataset])

    data_test = np.array([d[0] for d in test_dataset])
    Y_test = np.array([d[1] for d in test_dataset])
    Z_test = np.array([d[2] for d in test_dataset])

    clf = LogisticRegression(
        warm_start=True,
        penalty="l2",
        solver="sag",
        multi_class="multinomial",
        fit_intercept=True,
        verbose=10,
        max_iter=6,
        n_jobs=64,
        random_state=0,
        class_weight=None,
    )

    clf.fit(data_train, Z_train)
    print(f"Logistic Bias Acc.: {clf.score(data_test, Z_test)}")

    clf = LogisticRegression(
        warm_start=True,
        penalty="l2",
        solver="sag",
        multi_class="multinomial",
        fit_intercept=True,
        verbose=10,
        max_iter=6,
        n_jobs=64,
        random_state=0,
        class_weight=None,
    )

    clf.fit(data_train, Y_train)
    print(f"Logistic Profession Acc.: {clf.score(data_test, Y_test)}")

    mlp_clf = MLPClassifier(max_iter=100, verbose=10)
    mlp_clf.fit(data_train, Z_train)

    print(f"MLP Bias Acc.: {mlp_clf.score(data_test, Z_test)}")

    mdl = MDL(train_dataset)

    print(f"Bias MDL: {mdl.get_score(args, num_labels=2, label_id=2)}")
