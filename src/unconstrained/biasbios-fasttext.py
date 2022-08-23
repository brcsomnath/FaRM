#!/usr/bin/env python
# coding: utf-8
import os
import sys

sys.path.append("../")
sys.path.append("src/")

import warnings

import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch import nn, optim
from tqdm import tqdm, tqdm_notebook

warnings.filterwarnings("ignore")

import argparse
import pickle
from typing import Dict, List

import torch
from gensim.models import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from utils.loss import RateDistortionUnconstrained
from utils.net import Net
from utils.utils import *

STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now"
])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size',
                        type=int,
                        default=300,
                        help='eembedding_size ')
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device",
                        default="cuda:3",
                        type=str,
                        help="GPU device.")

    parser.add_argument("--dataset", default="bios-fasttext", type=str)
    parser.add_argument("--data_path",
                        default="data",
                        type=str)
    parser.add_argument("--encoded_path",
                        default="data/embeddings",
                        type=str)
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
                        help="Number of layers.")

    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="Training epochs.")
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


def load_word_vectors(fname):

    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.key_to_index.keys())
    return model, vecs, words


def get_embeddings_based_dataset(data: List[dict],
                                 word2vec_model,
                                 p2i,
                                 filter_stopwords=False):

    X, Y = [], []
    unk, total = 0., 0.
    unknown = []
    vocab_counter = Counter()

    for entry in tqdm(data, total=len(data)):

        y = p2i[entry["p"]]
        words = entry["hard_text_tokenized"].split(" ")
        if filter_stopwords:
            words = [w for w in words if w.lower() not in STOPWORDS]

        vocab_counter.update(words)
        bagofwords = np.sum([
            word2vec_model[w] if w in word2vec_model else word2vec_model["unk"]
            for w in words
        ],
                            axis=0)
        #print(bagofwords.shape)
        X.append(bagofwords)
        Y.append(y)
        total += len(words)

        unknown_entry = [w for w in words if w not in word2vec_model]
        unknown.extend(unknown_entry)
        unk += len(unknown_entry)

    X = np.array(X)
    Y = np.array(Y)
    print("% unknown: {}".format(unk / total))
    return X, Y, unknown, vocab_counter


def save_in_word2vec_format(vecs: np.ndarray, words: np.ndarray, fname: str):

    with open(fname, "w", encoding="utf-8") as f:

        f.write(str(len(vecs)) + " " + "300" + "\n")
        for i, (v, w) in tqdm_notebook(enumerate(zip(vecs, words))):

            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")


if __name__ == '__main__':
    args = get_parser().parse_args()
    set_seed(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train = load_dataset(f"{args.data_path}/biasbios/train.pickle")
    dev = load_dataset(f"{args.data_path}/biasbios/dev.pickle")
    test = load_dataset(f"{args.data_path}/biasbios/test.pickle")

    p2i, i2p = load_dictionary(
        f"{args.data_path}/biasbios/profession2index.txt")
    g2i, i2g = load_dictionary(f"{args.data_path}/biasbios/gender2index.txt")
    counter = count_profs_and_gender(train + dev + test)

    counter = count_profs_and_gender(train + dev + test)
    f, m = 0., 0.
    prof2fem = dict()

    for k, values in counter.items():
        f += values['f']
        m += values['m']
        prof2fem[k] = values['f'] / (values['f'] + values['m'])

    print(f / (f + m))
    print(prof2fem)

    word2vec, vecs, words = load_word_vectors(
        f"{args.data_path}/embeddings/crawl-300d-2M.vec")

    X_train, Y_train, unknown_train, vocab_counter_train = get_embeddings_based_dataset(
        train, word2vec, p2i)
    X_dev, Y_dev, unknown_dev, vocab_counter_dev = get_embeddings_based_dataset(
        dev, word2vec, p2i)
    X_test, Y_test, unknown_test, vocab_counter_test = get_embeddings_based_dataset(
        test, word2vec, p2i)

    vocab_bios, _ = list(zip(*vocab_counter_train.most_common(120000)))
    words_set = set(words)
    vocab_bios = [w for w in tqdm_notebook(vocab_bios) if w in words_set]
    vecs_for_vocab = np.array([word2vec[w] for w in tqdm_notebook(vocab_bios)])

    save_in_word2vec_format(vecs_for_vocab, vocab_bios,
                            f"{args.encoded_path}/vecs.vocab.bios.txt")
    word2vec_bios, _, _ = load_word_vectors(
        f"{args.encoded_path}/vecs.vocab.bios.txt")

    print("len train: {}; len dev: {}; len test: {}".format(
        len(train), len(dev), len(test)))
    mean_train = np.mean(X_train, axis=0, keepdims=True)
    mean_dev = np.mean(X_dev, axis=0, keepdims=True)
    mean_test = np.mean(X_test, axis=0, keepdims=True)

    Y_dev_gender = np.array([g2i[d["g"]] for d in dev])
    Y_test_gender = np.array([g2i[d["g"]] for d in test])
    Y_train_gender = np.array([g2i[d["g"]] for d in train])

    net = Net(args)
    net.to(device)

    LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

    train_dataset = encode(X_train, Y_train, Y_train_gender)
    dev_dataset = encode(X_dev, Y_dev, Y_dev_gender)
    test_dataset = encode(X_test, Y_test, Y_test_gender)

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

    itr = tqdm_notebook(range(args.epochs))

    for _, epoch in enumerate(itr, 0):
        total_loss = 0
        for step, (batch_embs, batch_y,
                   batch_z) in enumerate(train_dataloader):
            features = LN(net(batch_embs.to(device)))
            loss = criterion(args, features, batch_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            itr.set_description(
                f'Epoch = {epoch} Loss = {(total_loss / (step + 1)):.6f}')

        scheduler.step()

    if not os.path.exists(os.path.join(args.model_save_path, args.dataset)):
        os.makedirs(os.path.join(args.model_save_path, args.dataset))
    torch.save(net,
               f"{args.model_save_path}/{args.dataset}/net-{args.dataset}.pb")

    train_dataset = generate_debiased_embeddings(args, train_dataloader, net)
    dev_dataset = generate_debiased_embeddings(args, dev_dataloader, net)
    test_dataset = generate_debiased_embeddings(args, test_dataloader, net)

    data_train = np.array([d[0] for d in train_dataset])
    y_train = np.array([d[1] for d in train_dataset])
    z_train = np.array([d[2] for d in train_dataset])

    data_test = np.array([d[0] for d in test_dataset])
    y_test = np.array([d[1] for d in test_dataset])
    z_test = np.array([d[2] for d in test_dataset])

    clf = LogisticRegression(warm_start=True,
                             penalty='l2',
                             solver="sag",
                             multi_class='multinomial',
                             fit_intercept=True,
                             verbose=10,
                             max_iter=6,
                             n_jobs=64,
                             random_state=0,
                             class_weight=None)

    clf.fit(data_train, z_train)
    print(f"Bias Accuracy: {clf.score(data_test, z_test)}")

    mlp_clf = MLPClassifier(max_iter=20, verbose=10)
    mlp_clf.fit(data_train, y_train)
    mlp_clf.score(data_test, y_test)

    # Main Task Probing

    clf = LogisticRegression(warm_start=True,
                             penalty='l2',
                             solver="sag",
                             multi_class='multinomial',
                             fit_intercept=True,
                             verbose=10,
                             max_iter=6,
                             n_jobs=64,
                             random_state=0,
                             class_weight=None)

    clf.fit(data_train, y_train)
    print(f"Task Accuracy: {clf.score(data_test, y_test)}")
