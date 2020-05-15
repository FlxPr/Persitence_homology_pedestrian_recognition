import numpy as np
import pickle as pickle


def read_data(data_file=None):
    if data_file is None:
        data_file = "data/data_acc_rot.dat"
    try:
        f = open(data_file, "rb")
    except FileNotFoundError:
        raise FileNotFoundError('Could not find data file. Please fetch data_acc_rot.dat from '
                                'https://geometrica.saclay.inria.fr/team/Fred.Chazal/Projects/TDAPedestrian/'
                                'and place it at location {}'.format(data_file))

    data = pickle.load(f, encoding="latin1")
    f.close()

    data_a = data[0]
    data_b = data[1]
    data_c = data[2]
    label = data[3]

    return data_a, data_b, data_c, label


def read_persistence_diagram(index=1, file_path=None):
    if file_path is not None:
        return np.fromfile(file_path, sep='\n').reshape(-1, 3)
    elif index is not None:
        return np.fromfile('intermediary_data/persistence_diagrams/{}'.format(index), sep='\n').reshape(-1, 3)
    else:
        raise ValueError('Please provide either a valid file path or a diagram index')


def read_all_persistence_diagrams(max_index=300):
    return [read_persistence_diagram(diagram_index) for diagram_index in range(max_index)]


def read_similarity_matrix(dimension=0, file_path=None):
    if file_path is not None:
        matrix = np.genfromtxt(file_path, delimiter=',')
    elif dimension in (0, 1):
        matrix = np.genfromtxt('intermediary_data/similarity_matrices/{}_similarity_matrix.csv'.format(dimension),
                               delimiter=',')
    else:
        raise ValueError('Please specify dimension parameter (0 or 1) or valid file path')
    return matrix


def plot_point_clouds(indexes=(1, 31, 61, 101, 131, 161, 201, 231, 261), save=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    color = lambda label: 'red' if 'A' in str(label) else ('blue' if 'B' in str(label) else 'green')

    a, b, c, labels = read_data()
    fig = plt.figure()
    handles, labs = None, None

    for i, index in enumerate(indexes):
        time_series = (a + b + c)[index]
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        ax.scatter(time_series[:, 0], time_series[:, 1], time_series[:, 2], color=color(labels[index]), marker='o',
                   label=str(labels[index])[2], s=5)
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.zaxis.set_major_formatter(plt.NullFormatter())

        if i == 0:
            # trick to include legend to overall plot in the right order
            ax.scatter([], [], [], color='blue', label='B', s=5)
            ax.scatter([], [], [], color='green', label='C', s=5)
            handles, labs = ax.get_legend_handles_labels()

    fig.legend(handles, labs, loc='upper right', title='Pedestrian')
    plt.suptitle('Pedestrian acceleration point clouds on xyz axis')

    if save:
        plt.savefig('plots/pedestrian_point_clouds.png')

    plt.show()
