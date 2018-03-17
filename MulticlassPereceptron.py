import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MulticlassPerceptron():
    """
    Multiclass Perceptron class to classify multiclass data
    
    Functions:
        __init__: performs set up for the classifier of parameters
        fit: trains the classifier to learn weights to seperate data
        predict: will take in dataset and output the predictions, use after classifier is trained
        visualize: Can visualize your data and how the boundaries are formed in 2D graph.
    """
    def __init__(self, num_classes, num_features, classes):
        """
        Initializes the classifier with parameters suited for dataset
        :param num_classes: number of classes inside the data set
        :type num_classes: int
        :param num_features: dimension of the dataset (how many features there are)
        :type num_features: int
        :param classes: list of classes inside the dataset
        :type classes: list
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.classes = classes
        # Initialize weights and add bias term
        self.weights = {y: np.zeros(self.num_features) for y in self.classes}
        self.bias = {y: 0 for y in self.classes}
        self.clf = PCA(n_components=2)
    
    def fit(self, data_x, data_y, shuffle, stop=30):
        """
        Fits classifier to data provided with labels
        :param data_x: samples of data to pass in of shape [n_samples, n_features]
        :type data_x: numpy.ndarray
        :param data_y: labels for passed in data of shape [n_samples, 1]
        :type data_y: numpy.ndarray
        :param shuffle: whether to shuffle data per epoch
        :type shuffle: bool
        :param stop: how many iterations till termination if classifier doesn't improve
        :type stop: int
        :returns: None
        """
        weights = {**self.weights}
        # Add Bias
        data_x = np.insert(data_x, 0, 1, axis=1)
        for y in self.classes:
            weights[y] = np.insert(weights[y], 0, 0)
        keep_training = True
        stop_counter = 0
        min_errors = 0
        first_iter = True
        epoch = 1
        while keep_training:
            num_errors = 0
            for idx in range(len(data_x)):
                row = data_x[idx]
                label = data_y[idx]
                max_value, max_class = 0, 0
                for c in self.classes:
                    pred_value = np.dot(row, weights[c])
                    if pred_value >= max_value:
                        max_value = pred_value
                        max_class = c

                if not(max_class == label):
                    weights[label] += row
                    weights[max_class] -= row
                    num_errors += 1
            if shuffle == True:
                data = np.insert(data_x, 0, data_y, axis=1)
                np.random.shuffle(data)
                data_y = data[:,0]
                data_x = data[:,1:]
            if epoch % 2 == 0:
                print("Finished epoch %d.\tClassifier made %d errors." % (epoch, num_errors))
            epoch += 1
            if first_iter:
                min_errors = num_errors
                first_iter = False
            if num_errors < min_errors:
                print("New record for classifier! %d errors. Saving weights" % num_errors)
                for c in self.classes:
                    self.weights[c] = weights[c][1:]
                    self.bias[c] = weights[c][0]
                min_errors = num_errors
            if num_errors == 0:
                print("Fitted on data. Ending training. Final Result: %d errors" % num_errors)
                keep_training = False
            if num_errors > min_errors:
                stop_counter += 1
            else:
                stop_counter = 0
            if stop_counter == stop:
                print("Classifier hasn't improved in %d epochs. Stopping training. Min errors: %d" % (stop, min_errors))
                keep_training = False
                
    def predict(self, x):
        """
        Performs prediction on a data set
        :param x: data to run classifier on
        :type x: numpy.ndarray
        :return: predictions for the data
        :rtype: numpy.ndarray
        """
        predictions = np.zeros((len(x), 1))
        for i in range(len(x)):
            row = x[i,:]
            max_value, max_class = 0, 0
            for c in self.classes:
                pred_value = np.dot(row, self.weights[c]) + self.bias[c]
                if pred_value >= max_value:
                    max_value = pred_value
                    max_class = c
            predictions[i] = max_class
        return predictions
    
    def visualize(self, data_x, data_y, h=0.05):
        """
        :param data_x: data to visualize, if greater than 2 dimension, will perform PCA to reduce to 2-dim
        :type data_x: numpy.ndarray
        :param data_y: labels for data
        :type data_y: numpy.ndarray
        :param h: how many "meshes" to form
        :type h: float
        :return: None
        """
        if data_x.shape[1] > 2:
            data_x = self.clf.fit_transform(data_x)
        x1_min, x1_max = np.amin(data_x[:,0]) - 1, np.amax(data_x[:,0]) + 1
        x2_min, x2_max = np.amin(data_x[:,1]) - 1, np.amax(data_x[:,1]) + 1
        x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, h), 
                                       np.arange(x2_min, x2_max, h))
        preds = self.predict(np.c_[x1_mesh.ravel(), x2_mesh.ravel()])
        preds = preds.reshape(x1_mesh.shape)
        plt.figure(figsize=(6, 6), dpi= 80, facecolor='w', edgecolor='k')
        plt.contourf(x1_mesh, x2_mesh, preds, cmap=plt.cm.Paired)
        plt.title("Multiclass Perceptron Boundaries")
        for c in classes:
            indices = np.where(data_y == c)
            label = "Class " + str(c)
            plt.scatter(data_x[indices, 0], data_x[indices, 1], cmap=plt.cm.Accent, label=label)
        plt.legend()
