import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, n_classes, labels=None):
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred_labels):
        cm[true_label, pred_label] += 1

    try:
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(n_classes)
        if labels is None: labels = tick_marks
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()
    except:
        print("\nMatrice di Confusione:\n", cm)

    return cm