#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np

sys.path.append("../")
sys.path.append("src/")

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from torch import nn, optim
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
                        default=7,
                        type=int,
                        help="Number of layers.")
    parser.add_argument(
        "--embedding_size",
        default=300,
        type=int,
        help="Hidden size of the representation output of the text encoder.",
    )
    parser.add_argument("--device",
                        default="cuda:3",
                        type=str,
                        help="GPU device.")
    parser.add_argument(
        "--data_path",
        default="data/",
        type=str,
        help="Dataset path.",
    )
    parser.add_argument("--model_save_path",
                        default="saved_models/unconstrained/",
                        type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--ratio", default=0.5, type=float)
    parser.add_argument("--corruption_ratio", default=0.3, type=float)
    parser.add_argument("--corrupt_labels", default=False, type=bool)
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


def load_data(path, size, ratio=0.5):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    X, Y_p, Y_m = [], [], []
    n1 = int(size * ratio / 2)
    n2 = int(size * (1 - ratio) / 2)

    for fname, p_label, m_label, n in zip(fnames, protected_labels,
                                          main_labels, [n1, n2, n2, n1]):
        data = np.load(path + "/" + fname)[:n]
        for x in data:
            X.append(x)
        for _ in data:
            Y_p.append(p_label)
        for _ in data:
            Y_m.append(m_label)

    Y_p = np.array(Y_p)
    Y_m = np.array(Y_m)
    X = np.array(X)
    X, Y_p, Y_m = shuffle(X, Y_p, Y_m, random_state=0)
    return X, Y_p, Y_m


if __name__ == "__main__":
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_train, y_p_train, y_m_train = load_data(
        f"{args.data_path}/emoji_sent_race_{args.ratio}/train/",
        size=100000,
        ratio=args.ratio,
    )
    x_dev, y_p_dev, y_m_dev = load_data(
        f"{args.data_path}/emoji_sent_race_{args.ratio}/test/",
        size=100000,
        ratio=args.ratio,
    )

    if args.corrupt_labels:
        num_classes = max(y_p_train) + 1
        n_train = len(x_train)
        n_rand = int(len(x_train) * args.corruption_ratio)

        randomize_indices = np.random.choice(range(n_train),
                                             size=n_rand,
                                             replace=False)
        random_labels = np.random.choice(np.arange(num_classes),
                                         size=n_rand,
                                         replace=True)

        y_p_train[randomize_indices] = np.random.choice(np.arange(num_classes),
                                                        size=n_rand,
                                                        replace=True)

    X_train = [(d, label) for d, label in zip(x_train, y_p_train)]
    X_test = [(d, label) for d, label in zip(x_dev, y_p_dev)]

    train_dataloader = load(X_train, batch_size=args.batch_size)
    test_dataloader = load(X_test, batch_size=args.batch_size)

    net = Net(args)
    net.to(device)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    criterion = RateDistortionUnconstrained(gam1=args.gam1,
                                            gam2=args.gam2,
                                            eps=args.eps)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.mom,
                          weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100, 600], gamma=0.1)

    itr = tqdm(range(args.epochs))

    for _, epoch in enumerate(itr, 0):
        total_loss = 0
        for step, (batch_embs, batch_lbls) in enumerate(train_dataloader):
            features = LN(net(batch_embs.to(device)))
            loss = criterion(args, features, batch_lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            itr.set_description(
                f"Epoch = {epoch} Loss = {(total_loss / (step + 1)):.6f}")

        scheduler.step()

    torch.save(net, f"{args.model_save_path}/deepmoji_net_{args.ratio}.pb")

    train = [(d, label, bias)
             for d, bias, label in zip(x_train, y_p_train, y_m_train)]
    test = [(d, label, bias)
            for d, bias, label in zip(x_dev, y_p_dev, y_m_dev)]

    train_loader = load(train, batch_size=args.batch_size)
    test_loader = load(test, batch_size=args.batch_size)

    debiased_train = generate_debiased_embeddings(args, train_loader, net)
    debiased_test = generate_debiased_embeddings(args, test_loader, net)

    data_train = np.array([d[0] for d in debiased_train])
    y_label_train = np.array([d[1] for d in debiased_train])
    y_bias_train = np.array([d[2] for d in debiased_train])

    data_test = np.array([d[0] for d in debiased_test])
    y_label_test = np.array([d[1] for d in debiased_test])
    y_bias_test = np.array([d[2] for d in debiased_test])

    clf = MLPClassifier(max_iter=15, verbose=False)
    clf.fit(data_train, y_bias_train)

    print(f"Debiased Bias MLP Score: {clf.score(data_test, y_bias_test)}")

    clf = MLPClassifier(max_iter=6, verbose=False)
    clf.fit(data_train, y_label_train)

    print(f"Debiased Task MLP Score: {clf.score(data_test, y_label_test)}")

    debiased_classifier = LinearSVC(fit_intercept=True,
                                    class_weight="balanced",
                                    dual=False,
                                    C=0.1,
                                    max_iter=10000)
    debiased_classifier.fit(data_train, y_bias_train)

    print(
        f"Debiased Bias Linear Score: {debiased_classifier.score(data_test, y_bias_test)}"
    )

    debiased_classifier = LinearSVC(fit_intercept=True,
                                    class_weight="balanced",
                                    dual=False,
                                    C=0.1,
                                    max_iter=10000)

    debiased_classifier.fit(data_train, y_label_train)
    print(
        f"Debiased Task Linear Score: {debiased_classifier.score(data_test, y_label_test)}"
    )

    _, debiased_diffs = get_TPR(y_label_test,
                                debiased_classifier.predict(data_test),
                                y_bias_test)

    print(f"RMS: {rms(list(debiased_diffs.values()))}")
    print(
        f"Demographic Parity: {get_demographic_parity(debiased_classifier.predict(data_test), y_bias_test)}"
    )
