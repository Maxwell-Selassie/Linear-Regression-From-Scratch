import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class MyLinearRegression:
    def __init__(self,learning_rate: float = 0.1,epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
        self.weight = None
        self.bias = None
        self.error = None

    # -- training --
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features) #initalise weights
        self.bias = 0 #initalise bias

        for i in range(self.epochs):
            y_pred = X @ self.weight + self.bias #initial predictions
            self.error = y_pred - y #calculate errors

            #Gradient descent
            self.dw = (2/n_samples) * np.dot(X.T,self.error) #compute the derivative of the cost function with respect to weights
            self.db = (2/n_samples) * np.sum(self.error) #compute the derivative of the cost function with respect to the bias

            self.weight -= self.learning_rate * self.dw #update the weights
            self.bias -= self.learning_rate * self.db #update the bias

            mean_squared_error = np.mean(self.error**2) #calculate the mean squared error
            self.loss_history.append(mean_squared_error) #Log mean squared error

    # -- predictions
    def predict(self,X):
        y_pred = X @ self.weight + self.bias #prediction model
        return y_pred

    def get_loss_history(self):
        return self.loss_history #returns log history
    
    # -- Visualisations --
    def plot_loss_curve(self):
        plt.figure(figsize=(17,10))
        plt.plot(self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error')
        plt.title('Learning Curve')
        plt.show()
    
    def plot_fit(self, y_true, y_pred):
        plt.figure(figsize=(17,10))
        plt.plot(y_true, label="Actual", color="blue", alpha=0.7)
        plt.plot(y_pred, label="Predicted", color="red", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Values")
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.show()

    def scatterplot_fit(self, y_true, y_pred):
        plt.figure(figsize=(17,10))
        plt.scatter(range(len(y_true)), y_true, label="Actual", color="blue", alpha=0.7)
        plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color="red", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Values")
        plt.title("Actual vs Predicted (Scatter)")
        plt.legend()
        plt.show()

    def regplot_fit(self, y_true, y_pred):
        plt.figure(figsize=(17,10))
        sns.scatterplot(x=range(len(y_true)), y=y_true, label="Actual", color="blue", alpha=0.7)
        sns.scatterplot(x=range(len(y_pred)), y=y_pred, label="Predicted", color="red", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Values")
        plt.title("Actual vs Predicted (Seaborn)")
        plt.legend()
        plt.show()


        # -- metrics --
    def mean_absolute_error(self,y_true,y_pred):
        return np.mean(np.abs(y_true - y_pred)) #returns mean absolute error
    
    def root_mean_squared_error(self,y_true,y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) #returns root_mean_squared_error
    
    def r2_score(self,y_true,y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_total) #returns r2 score
