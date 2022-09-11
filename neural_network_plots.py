import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker


class Neural_Network_Performance:

    @staticmethod
    def confidence_figure(confidence_TP_MLP: np.ndarray, confidence_TN_MLP: np.ndarray) -> None:
        """
        Draw two figures and save them into files for future use.

        :param confidence_TP_MLP: array with true positive attempts
        :param confidence_TN_MLP: array with true negative attempts
        """
        # Number of true negatives vs number of true positives
        plt.figure()
        n_TP, bins_TP, patches_TP = plt.hist(confidence_TP_MLP, alpha=0.5, bins=200)
        n_TN, bins_TN, patches_TN = plt.hist(confidence_TN_MLP, alpha=0.5, bins=200)
        plt.legend(["score True Positive", "score True Negative"])
        plt.xlim([-1, 1])
        plt.xlabel("score")
        plt.grid()
        plt.savefig("./plots/confidence_TP_TN.png")
        plt.show()

        # Probability of true negatives based on the threshold
        plt.figure()
        plt.plot(bins_TP[1:], np.cumsum(n_TP) / np.sum(n_TP))
        plt.plot(bins_TN[1:], 1 - (np.cumsum(n_TN) / np.sum(n_TN)))
        plt.grid()
        plt.xlabel("threshold")
        plt.ylabel("probability")
        plt.savefig("./plots/threshold_probability.png")
        plt.show()

    @staticmethod
    def model_accuracy_figure(history) -> None:
        """
        Draw two figures about the accuracy and loss of model during training
        :param history: historical data of model training
        """
        # Model accuracy
        plt.figure()
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("./plots/model_accuracy.png")
        plt.show()

        # Model loss
        plt.figure()
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig("./plots/model_loss.png")
        plt.show()


