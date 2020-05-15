import numpy as np
import matplotlib.pyplot as plt
from persistence_diagrams import filter_persistence_diagram, plot_persistence_diagram, create_simple_persistence_diagram
from heapq import nlargest
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def _lambda_p(birth, death, t):
    if birth <= t <= (birth + death)/2:
        return t - birth
    elif (birth + death)/2 < t <= death:
        return death - t
    else:
        return 0


def _lambda_kt(nblandscapes, t, persistence_diagram):
    # Use heap search to retrieve nblandscapes first landscape values at x=t
    return nlargest(nblandscapes, (_lambda_p(birth, death, t) for birth, death in persistence_diagram))


def get_landscapes(persistence_diagram, dimension, xmin, xmax, nbnodes, nblandscapes=2):
    landscapes = [0] * nbnodes
    persistence_diagram = filter_persistence_diagram(persistence_diagram, dimension)
    for i, t in enumerate(np.linspace(xmin, xmax, nbnodes)):
        landscapes[i] = _lambda_kt(nblandscapes, t, persistence_diagram)
    return np.array(landscapes).transpose()


def plot_landscapes(landscapes, xmin=0, xmax=1, dimension=0, grid=True, save=True, title='landscape'):
    for i, landscape in enumerate(landscapes):
        plt.plot(np.linspace(xmin, xmax, len(landscape)), landscape, label=str(i+1))
    plt.xlim(xmin, xmax)
    plt.ylim(0, xmax)
    plt.title('{}d landscapes'.format(dimension))
    plt.gca().set_aspect('equal', adjustable='box')
    if grid:
        plt.grid()
    plt.legend(title='Landscape number')
    if save:
        plt.savefig('plots/{}.png'.format(title))
    plt.show()


def test_persistence_landscapes(xmin=0, xmax=1, save=True):
    persistence_diagram = create_simple_persistence_diagram()
    plot_persistence_diagram(persistence_diagram, title='test_persistence_diagram', save=save)
    lands_0 = get_landscapes(persistence_diagram, 0, xmin, xmax, 200, nblandscapes=3)
    lands_1 = get_landscapes(persistence_diagram, 1, xmin, xmax, 200, nblandscapes=4)

    plot_landscapes(lands_0, xmin, xmax, 0, True, save, 'test_persistence_landscape_0d')
    plot_landscapes(lands_1, xmin, xmax, 1, True, save, 'test_persistence_landscape_1d')


def plot_mean_landscapes_per_category(l_list, dimension=1, xmin=0, xmax=1.5, nbnodes=200,
                                      interval='std', filtration='rips', save=True):

    print('Plotting first {} mean landscapes in dimension {} with {} intervals'
          .format(l_list[0].shape[0], dimension, interval if interval else 'no'))

    category_slices = [slice(0, 100), slice(100, 200), slice(200, 300)]
    category_colors = ['red', 'blue', 'green']
    category_labels = ['A', 'B', 'C']

    for k in range(l_list[0].shape[0]):
        for cat_slice, color, label in zip(category_slices, category_colors, category_labels):
            category = l_list[cat_slice]
            category_mean = np.array([np.mean([landscape[k][i] for landscape in category]) for i in range(nbnodes)])

            if interval == 'std':
                category_std = np.array(
                    [np.std([landscape[k][i] for landscape in category]) for i in range(nbnodes)])
                category_lower = category_mean - category_std
                category_upper = category_mean + category_std

            elif interval == 'percentile':
                category_lower = [np.percentile([landscape[k][i] for landscape in category], 90) for i in
                                  range(nbnodes)]
                category_upper = [np.percentile([landscape[k][i] for landscape in category], 10) for i in
                                  range(nbnodes)]

            elif interval == 'bootstrap':
                category_bootstrap = [bs.bootstrap(np.array([landscape[k][i] for landscape in category]),
                                                   stat_func=bs_stats.std) for i in range(nbnodes)]
                category_lower = np.array([cat_mean_conf.lower_bound for cat_mean_conf in category_bootstrap])
                category_upper = np.array([cat_mean_conf.upper_bound for cat_mean_conf in category_bootstrap])
                category_mean = (category_lower + category_upper) / 2

            plt.plot(np.linspace(xmin, xmax, nbnodes), category_mean, color=color, label=label)

            if interval in ['bootstrap', 'percentile', 'std']:
                plt.fill_between(np.linspace(xmin, xmax, nbnodes), category_lower, category_upper, color=color, alpha=.1)

        plt.ylim(bottom=0)
        plt.legend(title='Pedestrian')
        interval_text = ', interval:{}'.format(interval) if interval in ['bootstrap', 'quantile', 'std'] else ''
        plt.title('{}-filtration {}d mean landscape {} {}'.format(filtration, dimension, k + 1, interval_text))
        if save:
            plt.savefig('plots/{}_{}d_mean_landscapes_{}_{}.png'.format(filtration, dimension, k + 1, interval))
        plt.show()


if __name__ == '__main__':
    test_persistence_landscapes()
