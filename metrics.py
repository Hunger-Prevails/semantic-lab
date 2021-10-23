import numpy as np

class Counter():

    def __init__(self, n_classes):
        self.n_classes = n_classes
        '''to override'''
        raise NotImplementedError()


    def update(self, gt, pred):
        '''to override'''
        raise NotImplementedError()


    def to_metric(self):
        '''to override'''
        raise NotImplementedError()


class ConfusionCounter(Counter):

    def __init__(self, n_classes):
        super().__init__(n_classes)
        self.confusion = np.zeros((n_classes, n_classes))


    def update(self, labels, predictions):
        matrices = [self.fast_hist(label.flatten(), prediction.flatten()) for label, prediction in zip(labels, predictions)]
        map(lambda matrix:self.confusion += matrix, matrices)


    def fast_hist(self, label, prediction):
        hist = np.bincount(self.n_classes * label.astype(np.int) + prediction, minlength = self.n_classes ** 2)
        return hist.reshape(self.n_classes, self.n_classes)


    def to_metrics(self):
        hist = self.confusion
        accuracy = np.diag(hist).sum() / hist.sum()
        recall_per_class = np.diag(hist) / hist.sum(axis = 1)
        overlap_per_class = np.diag(hist) / (hist.sum(axis = 1) + hist.sum(axis = 0) - np.diag(hist))
        class_density = hist.sum(axis = 1) / hist.sum()
        overlap_by_density = np.dot(class_density, overlap_per_class)

        return dict(
            accuracy = accuracy,
            recall_per_class = recall_per_class,
            overlap_per_class = overlap_per_class,
            overlap_by_density = overlap_by_density
        )
