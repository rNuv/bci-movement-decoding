from sklearn.svm import SVC

class SVMModel:
    """
    SVMModel represents the SVM classifier.
    """

    def __init__(self, alpha=1.0, kernel='linear'):
        """
        SVMModel constructor.

        @param alpha is the regularization strength.
        @param kernel is the type of kernel used in SVM.
        """

        self.model = SVC(C=alpha, kernel='rbf', probability=True)

    def train(self, A, D):
        """
        Fit the model with input matrix A with target D.

        @param A is the input feature matrix.∂
        @param D is the target vector.
        """

        self.model.fit(A, D)

    def predict(self, A):
        """
        Predict on new data A.

        @param A is new data input
        @return prediction vector
        """

        return self.model.predict(A)
    
    def weights(self):
        """
        Get SVM weights

        @return weights of the model.
        """

        return self.model.coef_[0]