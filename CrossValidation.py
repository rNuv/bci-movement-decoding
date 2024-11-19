from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from Visualization import *

class CrossValidation:
    """
    CrossValidation class performs a two-level nested cross-validation to find the optimal regularization value.
    """

    def __init__(self, svm_model, parameters, num_folds=6):
        """
        CrossValidation constructor.

        @param svm_model is the SVMModel used in CrossValidation.
        @param parameters is a list of regularizations strengths to test.
        @param num_folds is the number of folds to use in the first level.
        """

        self.svm_model = svm_model
        self.parameters = parameters
        self.num_folds = num_folds

    def split_folds(self, X, y):
        """
        Splits the dataset into stratified folds for cross-validation.

        @param X is the input data vector.
        @param y is the target vector.
        @return The indices of training/testing split
        """

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        return list(skf.split(X, y))
    
    def inner_cv(self, X, y):
        """
        Inner loop of the nested cross-validation to find the best regularization parameter.

        @param X is the input data vector.
        @param y is the target vector.
        @return Best regularization strength found from inner CV.
        """
        
        inner_cv = StratifiedKFold(n_splits=self.num_folds - 1, shuffle=True)
        clf = GridSearchCV(estimator=self.svm_model.model, param_grid=self.parameters, cv=inner_cv, scoring='accuracy')
        clf.fit(X, y)
        return clf.best_params_['C']
    
    def outer_cv(self, X, y, visualize_first_level=False, visualize_ROC=False):
        """
        Outer loop of the nested cross-validation to estimate the model performance.

        @param X is the input data vector.
        @param y is the target vector.
        @return avg accuracy of all outer folds, avg roc score of all outer folds, all regularization values, best regularization value
        """

        fold_indices = self.split_folds(X, y)
        accuracy_scores = []
        roc_auc_scores = []
        regularization_values = []

        fpr_list = []
        tpr_list = []
        roc_auc_list = []
        
        optimal_alpha = -1
        
        for train_index, test_index in fold_indices:
            print("Starting Outer fold")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # assert X_train.shape == (200,204), "X_train must be of shape (200,204)"
            # assert y_train.shape == (200,), "y_train must be of shape (200,)"
            # assert X_test.shape == (40,204), "X_test must be of shape (40,204)"
            # assert y_test.shape == (40,), "X_train must be of shape (40,)"

            # inner CV to find best regularization parameter
            cv_alpha = self.inner_cv(X_train, y_train)
            regularization_values.append(cv_alpha)
            
            # train and evaluate the model with the best alpha
            self.svm_model.model.C = cv_alpha
            self.svm_model.train(X_train, y_train)
            y_pred = self.svm_model.predict(X_test)
            y_prob = self.svm_model.model.predict_proba(X_test)[:, 1]
            # assert y_pred.shape == (40,), "y_pred must be of shape (40,)"
            # assert y_prob.shape == (40,), "y_prob must be of shape (40,)"
            
            # compute accuracy and roc_auc 
            accuracy = accuracy_score(y_test, y_pred) # NORMALIZATION SWITCHES PREDICTIONS?!?!
            roc_auc = roc_auc_score(y_test, y_prob)

            # compute metrics neccesary for roc plot
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(auc(fpr, tpr))
            
            accuracy_scores.append(accuracy)
            roc_auc_scores.append(roc_auc)
            
            # update optimal alpha (maximum accuracy)
            if accuracy == max(accuracy_scores):
                optimal_alpha = cv_alpha
            
            # visualize first level CV plots
            if visualize_first_level:
                visualize_channel_weights(self.svm_model)
                visualize_channel_stem_plot(self.svm_model)
        
        # visualize ROCs
        if visualize_ROC:
            visualize_ROC_plots(fpr_list, tpr_list, roc_auc_list)

        return np.mean(accuracy_scores), np.mean(roc_auc_scores), regularization_values, optimal_alpha