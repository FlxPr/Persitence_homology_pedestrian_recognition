import numpy as np
import matplotlib.pyplot as plt
from data_reader import read_data
import gudhi

max_edge_length = 2


def compute_persistence(edge_length=max_edge_length, filtration='rips'):
    if filtration not in ('rips', 'alpha'):
        raise ValueError('Please indicate filtration = "rips" or filtration = "alpha"')
    a, b, c, labels = read_data()
    time_series = a + b + c
    simplex_trees = [0] * len(time_series)

    print('Computing persistence diagrams with {} filtration...'.format(filtration))
    for i, ts in enumerate(time_series):
        if not i % 50:
            print('Computing persistence diagram {}/{}'.format(i, len(time_series)))

        if filtration is 'rips':
            cplx = gudhi.RipsComplex(points=ts, max_edge_length=edge_length)
            simplex_tree = cplx.create_simplex_tree(max_dimension=2)

        else:
            cplx = gudhi.AlphaComplex(points=ts)
            simplex_tree = cplx.create_simplex_tree()

        simplex_trees[i] = simplex_tree
        simplex_tree.persistence(persistence_dim_max=False)
        simplex_tree.write_persistence_diagram('intermediary_data/persistence_diagrams/{}'.format(i))

    return simplex_trees


def _to_numeric(array):
    return np.array([float(val) for val in array])


def filter_persistence_diagram(diagram, dimension):
    return [_to_numeric(triplet[1:]) for triplet in diagram if int(triplet[0]) == dimension]


def plot_persistence_diagram_with_gudhi(diagram_index=1):
    gudhi.plot_persistence_diagram(persistence_file='intermediary_data/persistence_diagrams/{}'.format(diagram_index),
                                   legend=True)
    plt.show()


def plot_persistence_diagram(persistence_diagram, hint_landscape=True, grid=True, save=True, title='persistence'):
    plt.plot([0, 1], [0, 1], color='black')

    persistence_0d = np.array(filter_persistence_diagram(persistence_diagram, dimension=0))
    persistence_1d = np.array(filter_persistence_diagram(persistence_diagram, dimension=1))

    plt.scatter(persistence_0d[:, 0], persistence_0d[:, 1], label='0', color='red')
    plt.scatter(persistence_1d[:, 0], persistence_1d[:, 1], label='1', color='blue')

    if hint_landscape:  # draw vertical and horizontal lines from points to diagonal line, showing landscapes
        for point in persistence_0d:
            plt.axvline(x=point[0], ymin=point[0], ymax=point[1], linestyle='dotted', color='red')
            plt.axhline(y=point[1], xmin=point[0], xmax=point[1], linestyle='dotted', color='red')
        for point in persistence_1d:
            plt.axvline(x=point[0], ymin=point[0], ymax=point[1], linestyle='dashed', color='blue')
            plt.axhline(y=point[1], xmin=point[0], xmax=point[1], linestyle='dashed', color='blue')

    if grid:
        plt.grid()

    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence diagram')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(title='Dimension')
    if save:
        plt.savefig('plots/{}.png'.format(title))
    plt.show()


def create_simple_persistence_diagram():
    return [
        [0, 0, 0.5],
        [0, 0, 0.6],
        [0, 0, 0.7],
        [1, 0.2, 0.6],
        [1, 0.4, 0.7],
        [1, 0.5, 0.8]
    ]
