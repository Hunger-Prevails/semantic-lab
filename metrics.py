import numpy as np

class Counter:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion = np.zeros((n_classes, n_classes), dtype = np.int32)


    def update(self, labels, predictions):
        for label, prediction in zip(labels, predictions):
            self.confusion += self.fast_hist(label.flatten(), prediction.flatten())


    def fast_hist(self, label, prediction):
        flag = label < self.n_classes

        indices = np.multiply(self.n_classes, label[flag]) + prediction[flag]

        hist = np.bincount(indices, None, np.power(self.n_classes, 2))

        return hist.reshape(self.n_classes, self.n_classes)


    def to_metrics(self):
        hist = self.confusion
        accuracy = np.diag(hist).sum() / hist.sum()
        recall_by_class = np.diag(hist) / (hist.sum(axis = 1) + 1e-6)
        overlap_by_class = np.diag(hist) / (hist.sum(axis = 1) + hist.sum(axis = 0) - np.diag(hist) + 1e-6)
        class_density = hist.sum(axis = 1) / hist.sum()
        overlap_by_density = np.dot(class_density, overlap_by_class)

        return dict(
            accuracy = accuracy,
            recall_by_class = recall_by_class.tolist(),
            overlap_by_class = overlap_by_class.tolist(),
            overlap_by_density = overlap_by_density
        )
