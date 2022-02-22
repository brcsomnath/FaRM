import math
from random import shuffle

import numpy as np
from sklearn.neural_network import MLPClassifier


class MDL:
    """
    Minimum Description Length (MDL)

    Computes the online code for MDL, implementing details provided
    in the paper -- https://arxiv.org/pdf/2003.12298.pdf

    Arguments:
        dataset: list of tuples [(x, y)], where x is a feature vector and y is the label
    """
    def __init__(self, dataset):
        super(MDL, self).__init__()
        shuffle(dataset)

        ratios = [
            0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5,
            1
        ]
        self.all_datasets = []

        for r in ratios:
            self.all_datasets.append(dataset[:int(r * len(dataset))])

    def get_score(self, args, num_labels, label_id=1):
        # bits required for the first transmission
        score = len(self.all_datasets[0]) * math.log(num_labels, 2)

        for i, dataset in enumerate(self.all_datasets[:-1]):
            X_train = np.array([x[args.DATA_IDX] for x in dataset])
            Y_train = [x[label_id] for x in dataset]

            clf = MLPClassifier()
            clf.fit(X_train, Y_train)

            next_dataset = self.all_datasets[i + 1]
            X_test = np.array([x[args.DATA_IDX] for x in next_dataset])
            Y_test = [x[label_id] for x in next_dataset]

            Y_pred = clf.predict_proba(X_test)

            for y_gold, y_pred in zip(Y_test, Y_pred):
                try:
                    score -= math.log(y_pred[y_gold], 2)
                except:
                    pass

            print(f"Iteration {i}: {score} bits")

        return (score / 1024)  # Final output in Kbits
