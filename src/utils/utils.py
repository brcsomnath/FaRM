import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel


def encode(X, Y, Z):
    """Converts data into token representations in the form of tensors."""
    encoded_dataset = []
    for x, y, z in zip(X, Y, Z):
        emb = torch.tensor(x)
        encoded_dataset.append((emb, y, z))
    return encoded_dataset


def load(dataset, batch_size, shuffle=True):
    """Loads the dataset using a dataloader with a batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_debiased_embeddings(args, dataset_loader, net):
    """
    Retrieve debiased embeddings post training 

    Arguments:
        args: arguments
        dataset_loader: pytorch data loader
        net: \phi(x) network
    
    Return:
        dataset: [debiased_embedding, y, z]
    """

    dataset = []
    for data, y, z in tqdm(dataset_loader):
        real_data = data.to(args.device)

        with torch.no_grad():
            output = net(real_data)

        purged_emb = output.detach().cpu().numpy()
        data_slice = [(data, int(y.detach().cpu().numpy()),
                       int(z.detach().cpu().numpy()))
                      for data, y, z in zip(purged_emb, y, z)]
        dataset.extend(data_slice)
    return dataset


def set_seed(args):
    """
    Set the random seed for reproducibility
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def load_dump(filename):
    """
    Util function to load a pickle file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def dump_data(filename, dataset):
    """
    Dumps data into .pkl file.
    """
    with open(filename, "wb") as file:
        pickle.dump(dataset, file)


def load_content(content, DATA_IDX=1, Y_IDX=2, Z_IDX=3):
    """
    Forms the data in the format (sentence, y-label, z-label)
    Returns an array of instances in the above format
    """
    dataset = []
    for c in content:
        dataset.append((c[DATA_IDX], c[Y_IDX], c[Z_IDX]))
    return dataset


def initialize_models(args, device):
    """
    Initialize a BERT model
    """

    bert_model = BertModel.from_pretrained(args.MODEL)
    bert_model.to(device)
    print(bert_model)

    return bert_model


def generate_purged_dataset_constrained(args, dataset_loader, bert_model):
    """
    Retrieve debiased embeddings post training in the constrained setup

    Arguments:
        args: arguments
        dataset_loader: pytorch data loader
        net: \phi(x) network
    
    Return:
        dataset: [debiased_embedding, y, z]
    """

    dataset = []
    for data, label, bias in tqdm(dataset_loader):
        real_data = data.to(args.device)

        with torch.no_grad():
            bert_output = bert_model(real_data)[1]

        purged_emb = bert_output.detach().cpu().numpy()
        data_slice = [(data, label, bias)
                      for data, label, bias in zip(purged_emb, label, bias)]
        dataset.extend(data_slice)
    return dataset


def generate_purged_dataset_multiple(args, dataset_loader, bert_model):
    """
    Retrieve debiased embeddings post training in the constrained setup
    while debiasing multiple protected attributes

    Arguments:
        args: arguments
        dataset_loader: pytorch data loader
        net: \phi(x) network
    
    Return:
        dataset: [debiased_embedding, y, z]
    """

    dataset = []
    for data, label, bias_1, bias_2 in tqdm(dataset_loader):
        real_data = data.to(args.device)

        with torch.no_grad():
            bert_output = bert_model(real_data)[1]

        purged_emb = bert_output.detach().cpu().numpy()
        data_slice = [
            (data, label, b1, b2)
            for data, label, b1, b2 in zip(purged_emb, label, bias_1, bias_2)
        ]
        dataset.extend(data_slice)
    return dataset


def get_TPR(y_main, y_hat_main, y_protected):
    """
    Computes the true positive rate (TPR)

    Arguments:
        y_main: main task labels
        y_hat_main: predictions for main task 
        y_protected: protected task labels

    Returns:
        diffs: different between TPRs
    """

    all_y = list(Counter(y_main).keys())

    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals["y:{}".format(label)]["p:{}".format(i)] = (
                y_label == y_hat_label).mean()

    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs


def get_demographic_parity(y_hat_main, y_protected):
    """
    Computes Demgraphic parity (DP)

    Arguments:
        y_hat_main: predictions for main task 
        y_protected: protected task labels

    Returns:
        dp: Demographic parity across all labels
    """

    all_y = list(Counter(y_hat_main).keys())

    dp = 0
    for y in all_y:
        D_i = []
        for i in range(2):
            used_vals = y_protected == i
            y_hat_label = y_hat_main[used_vals]
            Di_ = len(y_hat_label[y_hat_label == y]) / len(y_hat_label)
            D_i.append(Di_)
        dp += abs(D_i[0] - D_i[1])

    return dp


def rms(arr):
    """
    Computes the RMS value for a sequence of numbers
    """
    return np.sqrt(np.mean(np.square(arr)))


def evaluate_mlp(x_train, y_train, x_test, y_test):
    """
    Evaluates MLPScore for prediction on a task
    """

    clf = MLPClassifier()

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    F1 = f1_score(y_pred, y_test, average="micro") * 100
    P = precision_score(y_pred, y_test, average="micro") * 100
    R = recall_score(y_pred, y_test, average="micro") * 100
    return F1, P, R
