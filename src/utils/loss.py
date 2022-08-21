import numpy as np
import torch

# parts of the code have been adapted from https://github.com/ryanchankh/mcr2/blob/master/loss.py

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes."""
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.0
    return labels_onehot


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.0
    return Pi


class RateDistortion(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(RateDistortion, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def rate(self, args, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(args.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.0

    def rate_for_mixture(self, args, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(args.device)
        compress_loss = 0.0
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.0


class RateDistortionUnconstrained(RateDistortion):
    """
    Rate distortion loss in a unconstrained setup .
    """
    def forward(self, args, X, Z):
        W = X.T

        num_classes_z = Z.max() + 1
        Pi_z = label_to_membership(Z.numpy(), num_classes_z)
        Pi_z = torch.tensor(Pi_z, dtype=torch.float32).to(args.device)

        Rz_pi = self.rate_for_mixture(args, W, Pi_z)
        Rz = self.rate(args, W)

        J_u = -Rz - Rz_pi
        return J_u


class RateDistortionConstrained(RateDistortion):
    """
    Rate distortion loss in a constrained setup for
    debiasing a single protected attribute. 
    """
    def forward(self, args, X, Y):
        num_classes = Y.max() + 1

        W = X.T
        Pi = label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(args.device)

        R_z_pi = self.rate_for_mixture(args, W, Pi)
        R_z = self.rate(args, W)
        return R_z - R_z_pi, R_z, R_z_pi


class RateDistortionConstrainedMultiple(RateDistortion):
    """
    Computes the rate distortion function for debiasing
    multiple protected attributes simultaneously. (Section 7.2)

    1. 1-partition function
    2. $N$-partition function
    """
    def multiple_label_to_membership(self,
                                     Y_1,
                                     Y_2,
                                     num_classes_1=None,
                                     num_classes_2=None):
        """
        TODO: extend it to debias $N$ attributes
        """

        targets_1 = one_hot(Y_1, num_classes_1)
        targets_2 = one_hot(Y_2, num_classes_2)

        num_samples, _ = targets_1.shape
        Pi = np.zeros(shape=(num_classes_1 + num_classes_2, num_samples,
                             num_samples))

        for j in range(len(targets_1)):
            k = np.argmax(targets_1[j])
            Pi[k, j, j] = 0.5

            k = np.argmax(targets_2[j])
            Pi[num_classes_1 + k, j, j] = 0.5
        return Pi

    def one_partition(self, args, W, num_classes_1, num_classes_2, Y_1, Y_2):
        """
        TODO: extend it to debias $N$ attributes
        """

        Pi = self.multiple_label_to_membership(Y_1.numpy(), Y_2.numpy(),
                                               num_classes_1, num_classes_2)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(args.device)

        R_z_pi = self.rate_for_mixture(args, W, Pi)
        R_z = self.rate(args, W)

        return R_z - R_z_pi

    def n_partition(self, args, W, num_classes_1, num_classes_2, Y_1, Y_2):

        Pi_1 = label_to_membership(Y_1.numpy(), num_classes_1)
        Pi_1 = torch.tensor(Pi_1, dtype=torch.float32).to(args.device)

        Pi_2 = label_to_membership(Y_2.numpy(), num_classes_2)
        Pi_2 = torch.tensor(Pi_2, dtype=torch.float32).to(args.device)

        R_z_pi_1 = self.rate_for_mixture(args, W, Pi_1)
        R_z_pi_2 = self.rate_for_mixture(args, W, Pi_2)

        R_z = self.rate(args, W)
        return R_z - 0.5 * (R_z_pi_1 + R_z_pi_2)

    def forward(self, args, X, Y_1, Y_2):
        num_classes_1 = Y_1.max() + 1
        num_classes_2 = Y_2.max() + 1

        W = X.T

        if args.partition == "one":
            return self.one_partition(args, W, num_classes_1, num_classes_2,
                                      Y_1, Y_2)
        return self.n_partition(args, W, num_classes_1, num_classes_2, Y_1,
                                Y_2)
