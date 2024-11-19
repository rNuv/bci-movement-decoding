import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from ElectrodeDataset import ElectrodeDataset
from CrossValidation import CrossValidation
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from SVMModel import SVMModel
from Visualization import *

class MovementClassifier:
    """
    MovementClassifier is the entrypoint to the application.
    """

    def __init__(self, overt_files, imagined_files, transform_to_polar=False, normalize=False, use_pca=False):
        self.datasets = {
            "overt": [ElectrodeDataset(filepath=file, true_class=i%2, transform_to_polar=transform_to_polar, normalize=normalize, use_pca=use_pca) for i, file in enumerate(overt_files)],
            "imagined": [ElectrodeDataset(filepath=file, true_class=i%2, transform_to_polar=transform_to_polar, normalize=normalize, use_pca=use_pca) for i, file in enumerate(imagined_files)]
        }

    def train_and_test(self, train_data_type='overt', test_data_type='overt', alpha_values=[0.01, 0.1, 1, 10, 100, 1000], visualize_first_level=False, visualize_ROC=False):
        # load training data
        train_datasets = self.datasets[train_data_type]
        X_train = np.concatenate([ds.get_data().T for ds in train_datasets])
        y_train = np.concatenate([[ds.true_class] * ds.get_data().shape[1] for ds in train_datasets])

        # assert X_train.shape == (240, 204), "X_train must be of shape (204, 120)"
        # assert y_train.shape == (240,), "y_train must be of shape (204,)"

        # initialize SVM model
        svm_model = SVMModel(kernel='linear')

        # define parameter grid for cross-validation
        parameters = {'C': alpha_values}

        # initialize CrossValidation instance
        cv = CrossValidation(svm_model=svm_model, parameters=parameters, num_folds=6)

        # run nested cross-validation to get the performance metrics
        cv_accuracy, cv_roc_auc, regularization_values, optimal_alpha = cv.outer_cv(X_train, y_train, visualize_first_level=visualize_first_level, visualize_ROC=visualize_ROC)

        # if cross-training is required, retrain the model on all training data with the best alpha and test on the test set
        if train_data_type != test_data_type:
            # load testing data
            test_datasets = self.datasets[test_data_type]
            X_test = np.concatenate([ds.get_data().T for ds in test_datasets])
            y_test = np.concatenate([[ds.true_class] * ds.get_data().shape[1] for ds in test_datasets])

            # assert X_test.shape == (240, 204), "X_test must be of shape (204, 120)"
            # assert y_test.shape == (240,), "y_test must be of shape (204,)"

            # retrain on all training data with the best C from cross-validation
            svm_model.model.C = optimal_alpha
            svm_model.train(X_train, y_train)

            # predict on test data
            y_pred = svm_model.predict(X_test)
            # assert y_pred.shape == (240,), "y_pred must be of shape (204,)"

            # calculate accuracy and ROC AUC
            test_accuracy = accuracy_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_pred)

            return {
                "cross_validated_accuracy": cv_accuracy,
                "cross_validated_roc_auc": cv_roc_auc,
                "test_accuracy": test_accuracy,
                "test_roc_auc": test_roc_auc
            }
        else:
            # same-train case, return the cross-validated metrics
            return {
                "cross_validated_accuracy": cv_accuracy,
                "cross_validated_roc_auc": cv_roc_auc
            }

if __name__ == '__main__':
    overt_files = ["data/feaSubEOvert_1.csv", "data/feaSubEOvert_2.csv"]
    imagined_files = ["data/feaSubEImg_1.csv", "data/feaSubEImg_2.csv"]
    classifier = MovementClassifier(overt_files, imagined_files, transform_to_polar=False, normalize=False, use_pca=False)
    print(classifier.train_and_test(train_data_type='overt', test_data_type='imagined', visualize_first_level=False, visualize_ROC=False))