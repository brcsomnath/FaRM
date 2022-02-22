#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import sys

import torch

sys.path.append("../")
sys.path.append("src/")

from copy import deepcopy
from random import *

import numpy as np
import sklearn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import RateDistortionUnconstrained
from utils.mdl import MDL
from utils.net import Net
from utils.utils import *


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        help="Batch Size.")
    parser.add_argument("--num_layers",
                        default=4,
                        type=int,
                        help="Number of layers.")
    parser.add_argument(
        "--embedding_size",
        default=300,
        type=int,
        help="Hidden size of the representation output of the text encoder.")
    parser.add_argument("--DATA_IDX", default=0, type=int, help="Data ID")
    parser.add_argument("--device",
                        default="cuda:0",
                        type=str,
                        help="GPU device.")
    parser.add_argument("--data_path",
                        default="../data/embeddings",
                        type=str,
                        help="Dataset path.")
    parser.add_argument("--model_save_path",
                        default="saved_models/unconstrained/glove/",
                        type=str,
                        help="Save path of the models.")
    parser.add_argument("--epochs",
                        default=500,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--corrupt_labels", default=False, type=bool)
    parser.add_argument("--corruption_ratio", default=0.5, type=float)
    parser.add_argument("--truncate", default=False, type=bool)
    parser.add_argument("--truncation_ratio", default=0.8, type=float)
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--mom',
                        type=float,
                        default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd',
                        type=float,
                        default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--gam1',
                        type=float,
                        default=1.,
                        help='gamma1 for tuning empirical loss (default: 1.)')
    parser.add_argument('--gam2',
                        type=float,
                        default=1.,
                        help='gamma2 for tuning empirical loss (default: 1.)')
    parser.add_argument('--eps',
                        type=float,
                        default=0.5,
                        help='eps squared (default: 0.5)')
    return parser


def form_dataset(male_words, fem_words, neut_words):
    X, Y = [], []

    for w, v in male_words.items():
        X.append(v)
        Y.append(0)

    for w, v in fem_words.items():
        X.append(v)
        Y.append(1)

    for w, v in neut_words.items():
        X.append(v)
        Y.append(2)

    return np.array(X), np.array(Y)


def encode(X, Y):
    """Converts data into token representations in the form of tensors."""
    encoded_dataset = []
    for x, y in zip(X, Y):
        emb = torch.tensor(x)
        encoded_dataset.append((emb, y))
    return encoded_dataset


def dump_data(path, filename, dataset):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename), "wb") as file:
        pickle.dump(dataset, file)


def generate_debiased_embeddings(args, dataset_loader, net):
    dataset = []

    for data, label in tqdm(dataset_loader):
        real_data = data.to(args.device)

        with torch.no_grad():
            output = LN(net(real_data))

        purged_emb = output.detach().cpu().numpy()
        data_slice = [(data, int(label.detach().cpu().numpy()))
                      for data, label in zip(purged_emb, label)]
        dataset.extend(data_slice)
    return dataset


if __name__ == '__main__':

    args = get_parser().parse_args()

    assert not (args.truncate & args.corrupt_labels)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    male_words = load_dump(f'{args.data_path}/male_words.pkl')
    fem_words = load_dump(f'{args.data_path}/fem_words.pkl')
    neut_words = load_dump(f'{args.data_path}/neut_words.pkl')

    X, Y = form_dataset(male_words, fem_words, neut_words)

    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3, random_state=0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0)
    print("Train size: {}; Dev size: {}; Test size: {}".format(
        X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

    if args.truncate:
        train_X = X_train[:int(args.truncation_ratio * len(X_train))]
        train_Y = Y_train[:int(args.truncation_ratio * len(Y_train))]

    if args.corrupt_labels:
        Y_original = deepcopy(Y_train)

        num_classes = 3
        n_train = len(X_train)
        n_rand = int(len(X_train) * args.corruption_ratio)

        randomize_indices = np.random.choice(range(n_train),
                                             size=n_rand,
                                             replace=False)

        for idx in randomize_indices:
            label = np.random.choice(np.arange(num_classes))
            while label == Y_train[idx]:
                label = np.random.choice(np.arange(num_classes))
            Y_train[idx] = label

        assert not (Y_train[randomize_indices]
                    == Y_original[randomize_indices]).any()
        assert len(Y_original) == len(Y_train)

    net = Net(args)
    net.to(device)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    if args.truncate:
        train_dataset = encode(train_X, train_Y)
    else:
        train_dataset = encode(X_train, Y_train)

    dev_dataset = encode(X_dev, Y_dev)
    test_dataset = encode(X_test, Y_test)

    train_dataloader, dev_dataloader, test_dataloader = load(
        train_dataset,
        args.batch_size), load(dev_dataset,
                               args.batch_size), load(test_dataset,
                                                      args.batch_size)

    criterion = RateDistortionUnconstrained(gam1=args.gam1,
                                            gam2=args.gam2,
                                            eps=args.eps)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.mom,
                          weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400, 600], gamma=0.1)

    itr = tqdm(range(args.epochs))

    for _, epoch in enumerate(itr, 0):
        total_loss = 0
        for step, (batch_embs, batch_lbls) in enumerate(train_dataloader):
            features = LN(net(batch_embs.to(device)))
            loss = criterion(args, features, batch_lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss -= loss.item()
            itr.set_description(
                f'Epoch = {epoch} Loss = {(total_loss / (step + 1)):.6f}')

        scheduler.step()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if args.truncate:
        torch.save(
            net,
            f'{args.model_save_path}/word_emb_net-size-{args.truncation_ratio}.pb'
        )
    elif args.corrupt_labels:
        torch.save(
            net,
            f'{args.model_save_path}/word_emb_net-corrupt-{args.corruption_ratio}.pb'
        )
    else:
        torch.save(net, f'{args.model_save_path}/word_emb_net.pb')

    if args.truncate:
        train_dataset = encode(X_train, Y_train)
        train_dataloader = load(train_dataset, args.batch_size)

    if args.corrupt_labels:
        train_dataset = encode(X_train, Y_original)
        train_dataloader = load(train_dataset, args.batch_size)

    train_dataset = generate_debiased_embeddings(args, train_dataloader, net)
    dev_dataset = generate_debiased_embeddings(args, dev_dataloader, net)
    test_dataset = generate_debiased_embeddings(args, test_dataloader, net)

    data_train = np.array([d[0] for d in train_dataset])
    y_train = np.array([d[1] for d in train_dataset])

    data_test = np.array([d[0] for d in test_dataset])
    y_test = np.array([d[1] for d in test_dataset])

    data_train = data_train[y_train != 2]
    y_train = y_train[y_train != 2]

    data_test = data_test[y_test != 2]
    y_test = y_test[y_test != 2]

    F1, P, R = evaluate_mlp(data_train, y_train, data_test, y_test)
    print(f"Gender: F1 - {F1} P - {P} R - {R} ")

    dataset = [(x, y) for x, y in zip(data_train, y_train)]
    mdl = MDL(dataset)

    print(f"MDL: {mdl.get_score(args, num_labels=2)}")

    rank_before = np.linalg.matrix_rank(X_train)
    rank_after = np.linalg.matrix_rank(data_train)

    print("Rank before: {}; Rank after: {}".format(rank_before, rank_after))
