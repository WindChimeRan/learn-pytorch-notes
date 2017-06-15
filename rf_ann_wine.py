import pandas as pd
import os
import numpy as np


batch_size = 10
steps = 1000

# data
if not os.path.exists('wine/wine.data'):
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
else:
    df_wine = pd.read_csv('wine/wine.data')

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df_wine = df_wine.dropna(axis=0)
X = df_wine[df_wine.columns].values
y = df_wine['Class label'].values

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,  random_state=1)

data = (X_train, X_test, y_train, y_test )


def nn_classifier(data):

    import torch
    from torch import nn, optim
    from torch.autograd import Variable
    X_train, X_test, y_train, y_test = data
    X_train = np.array(X_train,dtype=np.float32)
    X_test = np.array(X_test,dtype=np.float32)
    y_train = Variable(torch.from_numpy(y_train)).cuda()
    X_train = Variable(torch.from_numpy(X_train)).cuda()
    y_test = Variable(torch.from_numpy(y_test)).cuda()
    X_test = Variable(torch.from_numpy(X_test)).cuda()

    classifier = nn.Sequential(
        nn.Linear(14,28),
        nn.Tanh(),
        nn.Linear(28, 28),
        nn.Tanh(),
        nn.Linear(28, 3),
        nn.Sigmoid()
    )


    classifier.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(),lr=1e-3)

    def data_loader(dataset):
        x,y = dataset
        def data_batch(batch_size,steps):

            for i in range(steps):
                t = np.random.randint(0, 104 - batch_size)
                yield x[t:batch_size+t,:],y[t:batch_size+t]
        return data_batch

    train_data = data_loader((X_train,y_train))

    for i,(x,y) in enumerate(train_data(batch_size,steps)):

        out = classifier(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    classifier.eval()
    out = classifier(X_test)
    _, pred = torch.max(out, 1)
    num_correct = (pred == y_test).sum()
    running_acc = (num_correct.data[0]+0.0)/X_test.size()[0]
    print "nn ",running_acc


def rf(data):
    X_train, X_test, y_train, y_test = data
# # random forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
    rf = rf.fit(X_train, y_train)

    # inference
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    #estimate
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)


    print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
if __name__ == '__main__':
    nn_classifier(data)
    rf(data)