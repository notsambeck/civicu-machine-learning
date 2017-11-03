import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import tqdm

df = pd.read_csv('../lessons/shared-resources/heights_weights_genders.csv')

df.Gender = (df.Gender == 'Male').astype(int)
male = df.Gender == 1

X = df[['Height', 'Weight']].values
y = df.Gender.values
w = np.random.randn(X.shape[1] + 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_proba(X, w):
    '''
    predict genders of array of normalized [Height, Weight]
    '''
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    raw_pred = np.dot(w, X.T)
    prob = sigmoid(raw_pred)

    return prob


def total_error(y, y_prob):
    '''
    return loss as a sum
    each y[i] is 0 or 1
    if y[i] is 0, add to error y_prob
    if y[i] is 1, add to error 1 - y_prob
    '''
    e = abs(np.dot(y_prob, np.subtract(y, 1))) + \
        abs(np.dot(np.subtract(y_prob, 1), y))
    return e / len(y)


def accuracy(y, y_prob):
    '''return percentage of samples that sigmoid predicts accurately'''
    assert len(y) == len(y_prob)
    pred = y_prob > .5
    return sum(y == pred) / len(y)


def dumb_gradient(w, X, y, step):
    prob = predict_proba(X, w)
    error = total_error(y, prob)
    acc = accuracy(y, prob)
    # print('error @ epoch start: {}; accuracy: {}'.format(error, acc))
    grad = []
    # print('input weights: {}: {}'.format(type(w), w))
    for i in range(len(w)):
        new_w = w.copy()
        new_w[i] += step
        # print(new_w)
        new_error = total_error(y, predict_proba(X, new_w))
        grad.append((error - new_error))

    # grad is the change in error when w[i] is increased by .01
    # grad = dE/dw for w in weights
    # if grad[i] is positive,
    # print('unnormalized grad: {:.5} {:.5} {:.5}'.format(*[g for g in grad]))

    return np.array(grad), error, acc


class Logistic():
    def __init__(self, X, y):
        print('building logistic regressor on X, y')
        self.X = X.copy()
        self.y = y.copy()
        self.n = X.shape[0]
        self.step = .001
        self.rate = 10000

        self.w = np.random.randn(X.shape[1] + 1)
        print('initialized weights:', self.w)

        self.err_history = []
        self.acc_history = []

        self.means = []
        self.stds = []
        self.standardize()

        for i in range(1):
            print('a sample:', self.X[np.random.randint(0, self.n - 1)])

    def standardize(self):
        for i in range(self.X.shape[1]):
            # save mean, std so new values can be converted/standardize values
            self.means.append(self.X[:, i].mean())
            print('mean of i=X[{}]: {}'.format(i, self.means[i]))
            self.X[:, i] = np.subtract(self.X[:, i], self.means[i])
            self.stds.append(self.X[:, i].std())
            print('std dev of X[{}]: {}'.format(i, self.stds[i]))
            self.X[:, i] = np.divide(self.X[:, i], self.stds[i])     # / std
        print('data normalized')

    def train_(self, v=1):
        grad, error, acc = dumb_gradient(self.w, self.X, self.y, self.step)
        if v:
            print('weights are: {:f} {:f} {:f}'.format(*self.w))
            print('gradient is: {:f} {:f} {:f}'.format(*grad))
            print('error: {:.5} accuracy: {:.2}'.format(error, acc))

        self.w = np.add(np.multiply(grad, self.rate), self.w)

        self.err_history.append(error)
        self.acc_history.append(acc)

    def predict_new_sample(self, x):
        '''predict new array of [heights, weights]'''
        x = np.subtract(x, self.means)
        x = np.divide(x, self.stds)
        return predict_proba(np.array(x).reshape(-1, 2), self.w)

    def train(self, n):
        for i in tqdm.tqdm(range(n)):
            self.train_(v=0)

        plt.scatter(x=range(len(self.err_history)), y=self.err_history,
                    s=1, c='r')
        plt.scatter(x=range(len(self.acc_history)), y=self.acc_history,
                    s=1, c='g')
        plt.ylabel('loss in red; acc in green')
        plt.show()

    def mask_plot(self, df):
        pred_male = self.predict_new_sample(df[['Height', 'Weight']]) > .5
        print((pred_male == y).sum())

        ax = df[pred_male].plot.scatter(x='Height', y='Weight', s=2,
                                        alpha=.3)
        ax = df[~pred_male].plot.scatter(ax=ax, x='Height', y='Weight', s=2,
                                         alpha=.3, c='red')
        ax = df[male].plot.scatter(ax=ax, x='Height', y='Weight', s=.1,
                                   alpha=1, c='k')
        ax = df[~male].plot.scatter(ax=ax, x='Height', y='Weight', s=.1,
                                    alpha=1, c='yellow')
        plt.show()


if __name__ == '__main__':
    lr = Logistic(X, y)
    lr.train(250)
    lr.mask_plot(df)
