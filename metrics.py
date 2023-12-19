import numpy as np

"""
Copy paste from the EasyPortrait repo
"""
class SegMetric(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """
        Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        FP = hist.sum(axis=0) - np.diag(hist)
        FN = hist.sum(axis=1) - np.diag(hist)
        TP = np.diag(hist)
        # TN = hist.sum() - (FP + FN + TP)
        epsilon = 1e-6
        precision = TP / (TP+FP+epsilon)
        recall = TP / (TP+FN+epsilon)
        f1 = (2 * (precision*recall) / (precision + recall + epsilon)).mean()

        iu = np.diag(hist) / (hist.sum(axis=1) +
                              hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Mean IoU : \t": mean_iu,
                "Overall F1: \t": f1,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))