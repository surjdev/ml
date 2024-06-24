import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        #compute the distrance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
# *********************************************************************************************************************************************************
class LinearRegression:

    def __init__(self,lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db




    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

# *********************************************************************************************************************************************************

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self,lr = 0.001,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            linear_pred = np.dot(X, self.weight) + self.bias
            pred = sigmoid(linear_pred)

            dw = (np.dot(X.T,(pred - y)))/n_samples
            db = np.sum((pred-y))/n_samples

            self.weight = self.weight - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weight) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

#***********************************************************************************
class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *,value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
#***********************************************************************************
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self,X,y):
        self.n_feature = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X,y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)


        # create child nodes
        left_idxs,right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs], depth+1)
        return Node(best_feature,best_thresh,left,right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        if not isinstance(feat_idxs, list):
            feat_idxs = [feat_idxs]

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._information_gain(y,X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold
    
    def _information_gain(self,y,X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        
        # create children
        left_idxs,right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted entropy of children
        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/ len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])
        
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
#***********************************************************************************
class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #calculate mean var and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        #calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        #return class with the highest posterior
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean) ** 2)/(2*var))
        denominator = np.sqrt(2* np.pi*var)
        return numerator/denominator
    
#***********************************************************************************

