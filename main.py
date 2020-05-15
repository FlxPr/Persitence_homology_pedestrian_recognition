from data_reader import read_all_persistence_diagrams, plot_point_clouds
from persistence_diagrams import compute_persistence
from bottleneck_similarity_matrix import compute_similarities
from persistence_landscapes import get_landscapes, test_persistence_landscapes, plot_mean_landscapes_per_category
from random_forest import train_random_forest, train_random_forest_on_raw_data
from random import seed
import time
import json


# Reproducibility
seed(1)


def run_experiment(save_results=True, filtration='rips', xmin=0, xmax=1.5, nbnodes=200, n_landscapes=5,
                   compute_similarity_matrices=True, max_edge_length=2, rforest_hyperparams={}):
    results = {'experiment_params': {'xmin': xmin,
                                     'xmax': xmax,
                                     'nbnodes': nbnodes,
                                     'n_landscapes': n_landscapes,
                                     'max_edge_length': max_edge_length,
                                     'filtration':filtration}}

    # Compute persistence diagrams
    tic = time.time()
    simplex_trees = compute_persistence(filtration=filtration, edge_length=max_edge_length)
    toc = time.time()
    results['persistence_computing_time_minutes'] = (toc - tic)/60
    print('Persistence diagram computation took {:.2f} minutes'.format(results['persistence_computing_time_minutes']))

    # Compute similarity matrices
    if compute_similarity_matrices:  # Set to False to avoid lengthy computation
        compute_similarities(simplex_trees, save=save_results, filtration=filtration)
    else:
        print('Skipped computing similarity matrices and plotting diagram similarities.\n'
              'Set compute_similarity_matrices to True to compute and save matrices and plots')

    # Compute persistence landscapes
    print('Computing persistence landscapes...')
    persistence_diagrams = read_all_persistence_diagrams()

    l0_list = [get_landscapes(persistence_diagram, 0, xmin, xmax, nbnodes, n_landscapes)
               for persistence_diagram in persistence_diagrams]
    l1_list = [get_landscapes(persistence_diagram, 1, xmin, xmax, nbnodes, n_landscapes)
               for persistence_diagram in persistence_diagrams]

    plot_mean_landscapes_per_category(l0_list, dimension=0, filtration=filtration, xmin=xmin, xmax=xmax,
                                      save=save_results, interval='bootstrap', nbnodes=nbnodes)
    plot_mean_landscapes_per_category(l1_list, dimension=1, filtration=filtration, xmin=xmin, xmax=xmax,
                                      save=save_results, interval='bootstrap', nbnodes=nbnodes)

    # Train random forest
    classification_results = train_random_forest(l0_list, l1_list, filtration=filtration, save=save_results,
                                                 hyperparams=rforest_hyperparams)
    results.update(classification_results)

    if save_results:
        with open('classification_results/{}.json'.format(filtration), 'wt') as f:
            json.dump(results, f)

    return results


def run_all_experiments(param_dict_list):
    plot_point_clouds(save=True)
    test_persistence_landscapes(save=True)

    results = []
    for i, param_dict in enumerate(param_dict_list):
        print('Running pipeline with parameters :\n{}'.format(param_dict))
        result = run_experiment(**param_dict)
        result['params'] = param_dict
        results.append(results)
        print("Results : \n{}".format(results[i]))

    raw_data_results = train_random_forest_on_raw_data(save=True)
    raw_data_results['params'] = 'Training on raw data'
    results.append(raw_data_results)

    return results


if __name__ == '__main__':
    experiments = [
        {
            'filtration': 'rips',
            'xmin': 0,
            'xmax': 1.5,
            'max_edge_length': 2,
            'nbnodes': 100,
            'compute_similarity_matrices': True,
            'save_results': True,
            'rforest_hyperparams': {'n_estimators': 150, 'max_features': 'auto'}
        },
        {
            'filtration': 'alpha',
            'xmin': 0,
            'xmax': 0.8,
            'nbnodes': 100,
            'compute_similarity_matrices': True,
            'save_results': True,
            'rforest_hyperparams': {'n_estimators': 150, 'max_features': 'auto'}
        }
    ]

    experiment_results = run_all_experiments(experiments)
