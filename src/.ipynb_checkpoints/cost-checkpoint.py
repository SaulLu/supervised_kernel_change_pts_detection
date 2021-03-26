import itertools
import random

import numpy as np
import ruptures as rpt
import tqdm


class KernelWithPartialAnnotationCost(rpt.base.BaseCost):
    """"""

    model = "Learned Kernel Partial Annotation"
    min_size = 3

    def pre_fit(
        self,
        signals,
        labels,
        initial_kernel_fct,
        upper_bound_similarity,
        lower_bound_dissimilarity,
        gamma,
    ):
        """computes the parameters (G_hat, G, training_samples) of the learned metrics

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must integers (>=0).
            upper_bound_similarity: #TODO
            lower_bound_dissimilarity: #TODO
            gamma: #TODO
        """
        self.initial_kernel_fct = initial_kernel_fct
        self.u = upper_bound_similarity
        self.l = lower_bound_dissimilarity
        self.gamma = gamma

        self.training_samples, self.constrains = self.get_training_samples_and_constains(signals, labels)

        self.G = initial_kernel_fct(self.training_samples, self.training_samples)

        self.G_hat = self.compute_bregman()

        self.G_inv = np.linalg.inv(self.G)  # Problem HERE

        # print(f"self.G_inv @ self.G: {self.G_inv @ self.G}, must be equal to the identity")

        self.G_core = self.G_inv @ (self.G_hat - self.G) @ self.G_inv

    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        Returns:
            self
        """
        self.signal = signal

    def error(self, start, end):
        """"""
        # compute equation 8.7) and then 8.8) in Charles Truong. Détection de ruptures multiples –
        # application aux signaux physiologiques.

        subsignal = self.signal[start:end]

        self.inner_product = self.initial_kernel_fct(subsignal, subsignal)

        self.inner_product_with_training_samples = self.initial_kernel_fct(subsignal, self.training_samples)

        # TODO: optimisation replace np.diag(self.initial_kernel_fct(subsignal, subsignal)) by
        # self.initial_kernel_fct.diag(subsignal)
        inner_product_sum = np.sum(np.diag(self.inner_product)) - 1.0 / (end - start) * np.sum(self.inner_product)
        second_term = (
            self.inner_product_with_training_samples @ self.G_core @ self.inner_product_with_training_samples.T
        )
        new_kernel_product = np.sum(np.diag(second_term)) - 1.0 / (end - start) * np.sum(second_term)
        cost_bis = inner_product_sum + new_kernel_product

        return cost_bis

    def _phi_m_hat_phi(self, i, j):
        ki = self.inner_product_with_training_samples[i, :][np.newaxis, :]
        kj = self.inner_product_with_training_samples[j, :][np.newaxis, :]

        return self.inner_product[i, j] + (ki @ self.G_core @ kj.T)[0][0]

    @staticmethod
    def get_training_samples_and_constains(signals, labels):
        # TODO: Handle (or not?) the duplicate in training_samples

        training_samples = []
        constrains = {}
        last_idx = -1
        for signal, label in zip(signals, labels):
            begin_signal = True
            idx_max = np.max(label)
            for idx in range(idx_max + 1):

                sub_signal = signal[(label == idx).squeeze()]
                new_idx_iterator = range(last_idx + 1, sub_signal.shape[0] + last_idx + 1)

                for key in itertools.combinations(new_idx_iterator, r=2):
                    constrains[key] = 1  # similar

                if begin_signal:
                    begin_signal = False
                else:
                    for key in itertools.product(last_idx_iterator, new_idx_iterator):
                        constrains[key] = -1  # disimilar

                training_samples.append(sub_signal)
                last_idx_iterator = new_idx_iterator
                *_, last_idx = last_idx_iterator

        training_samples = np.concatenate(training_samples)
        print(f"training_samples.shape: {training_samples.shape}")

        return training_samples, constrains

    def compute_bregman(self):

        eps = np.finfo(float).eps

        K = self.G.copy()
        lambdas = dict.fromkeys(self.constrains.keys(), np.array(0))
        xi = dict([(key, self.u) if value == 1 else (key, self.l) for key, value in self.constrains.items()])

        n = K.shape[0]
        convergence = True

        min_ite = int(len(self.constrains)/16)
        num_ite = 0

        while convergence:
            num_ite += 1
            print(f"Iteration n°: {num_ite}")
            convergence = False
            for _ in tqdm.tqdm(range(min_ite), position=0, leave=True):
                (i, j), delta = random.choice(list(self.constrains.items()))

                if i != j:

                    ei = np.zeros((n, 1))
                    ei[i] = 1
                    ej = np.zeros((n, 1))
                    ej[j] = 1

                    p = ((ei - ej).T @ K @ (ei - ej))[0][0]

                    alpha = np.minimum(
                        lambdas[(i, j)],
                        delta * self.gamma * (1.0 / (p + eps) - 1.0 / (xi[(i, j)] + eps)) / (self.gamma + 1.0),
                    )
                    beta = delta * alpha / (1 - delta * alpha * p)
                    xi[(i, j)] = self.gamma * xi[(i, j)] / (self.gamma + delta * alpha * xi[(i, j)])
                    lambdas[(i, j)] = lambdas[(i, j)] - alpha

                    update = beta * (K @ (ei - ej) @ (ei - ej).T @ K)

                    K = K + update

                    if np.linalg.norm(update) > 1e-6:
                        # the convergence is detected when there is a minimum number of iteration
                        # during which the upate is below a minimal value
                        convergence = True

        return K
