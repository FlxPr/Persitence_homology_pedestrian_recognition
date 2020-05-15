import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_reader import read_data
import json


def train_random_forest(l0_list, l1_list, hyperparams={}, save=False, filtration='rips', nbnodes=200):
    l0_list = [landscapes.flatten() for landscapes in l0_list]
    l1_list = [landscapes.flatten() for landscapes in l1_list]
    l1l2_list = [np.append(landscapes_0, landscapes_1) for landscapes_0, landscapes_1 in zip(l0_list, l1_list)]

    trainsets = [l0_list, l1_list, l1l2_list]
    trainset_names = ['l0', 'l1', 'l0 + l1']

    label = ['A'] * 100 + ['B'] * 100 + ['C'] * 100

    results = {}
    for train_set, train_set_name in zip(trainsets, trainset_names):
        avg = 0
        for i in range(20):
            l_train, l_test, label_train, label_test = train_test_split(train_set, label, test_size=0.2)
            rf = RandomForestClassifier(**hyperparams)
            rf.fit(np.array(l_train), label_train)
            avg += np.mean(rf.predict(l_test) == label_test)

        print("Random forest average prediction with {}: ".format(train_set_name), avg/20)
        results[train_set_name] = avg/20

        rf = RandomForestClassifier(**hyperparams)
        rf.fit(train_set, label)
        plot_feature_importance(rf, train_set_name, save=save, filtration=filtration, nbnodes=nbnodes)

    results['rforest_hyperparameters'] = hyperparams
    return results


def train_random_forest_on_raw_data(hyperparams={}, save=True):
    a, b, c, labels = read_data()
    time_series = a + b + c
    time_series = [ts.flatten() for ts in time_series]

    avg = 0
    for i in range(20):
        l_train, l_test, label_train, label_test = train_test_split(time_series, labels, test_size=0.2)
        rf = RandomForestClassifier(**hyperparams)
        rf.fit(np.array(l_train), label_train)
        avg += np.mean(rf.predict(l_test) == label_test)

    print("avg pred on raw data: ", avg / 20)
    results = {'accuracy': avg/20, 'rforest_hyperparameters': hyperparams}

    if save:
        with open('classification_results/raw_data_results.json', 'wt') as f:
            json.dump(results, f)

    rf = RandomForestClassifier(**hyperparams)
    rf.fit(time_series, labels)
    plot_feature_importance(rf, 'raw data', save=True, filtration='')

    return results


def plot_feature_importance(rf, trainset_name, save=False, filtration='rips', nbnodes=200):
    plt.plot(rf.feature_importances_, color='blue')
    plt.xlabel('Feature index')
    plt.ylabel('Feature importance')

    if trainset_name is not 'raw data':  # Plot landscape separation
        for i in range(1, len(rf.feature_importances_)//nbnodes):
            plt.axvline(x=i * nbnodes, linestyle='--', color='red')

    plt.title('Random forest feature importances for {} trainset \n of {} filtration'
              .format(trainset_name, filtration, 'with {} filtration'.format(filtration) if filtration else None))
    if save:
        plt.savefig('plots/{}_rforest_feature_importances_{}.png'.format(filtration, trainset_name))

    plt.show()
