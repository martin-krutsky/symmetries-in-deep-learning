import numpy as np
from sklearn.decomposition import PCA
from matplotlib import rc
# activate latex text rendering
rc('text', usetex=True)
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from weight_init import init_weight_grid
from neural_net import L_model_forward, compute_cost_entropy


def compute_single_err(X, y, weight_comb, act_function='relu'):
    parameters = dict()
    for l, layer in enumerate(weight_comb):
        # print('layer:', layer)
        parameters['W' + str(l+1)] = layer[:, :-1]
        parameters['b' + str(l + 1)] = layer[:, -1]

    pred = L_model_forward(X, parameters, hidden_act=act_function)[0]
    return compute_cost_entropy(pred, y)


def visualize_3D_err_space_all_possibilities(X, y, dim_list, nr_of_points_in_dir, act_function='relu', biases=True, sym_dim=None, sym_weights=None, name=None, xyzlabel=None):
    if sym_dim:
        weight_combs = init_weight_grid(sym_dim, -1, 1, nr_of_points_in_dir, biases=biases, sym_weights=sym_weights)
    else:
        weight_combs = init_weight_grid(dim_list, -1, 1, nr_of_points_in_dir, biases=biases)
    visualize_3D_err_space(X, y, weight_combs, act_function=act_function, biases=biases, indexes_to_keep=[0, 2], name=name, save_to_all=True, dont_show=True, xyzlabel=xyzlabel)


def calc_cross_flat_xor(X, y, weight_combs, i, act_function='relu'):
    cross_entropies = []
    flattened_ls = []
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
        cross_entropies.append(compute_single_err(X, y, comb, act_function=act_function))
    
    np.savez('xor_cross_flat{}.npz'.format(i), flat=flattened, cross=cross_entropies)
    
    
def visualize_3D_err_XOR(X, y, weight_combs, act_function='relu', biases=True, name=None, xyzlabel=None, zlabelpad=None):
    cross_entropies = []
    flattened_ls = []
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
        cross_entropies.append(compute_single_err(X, y, comb, act_function=act_function))
    
    if biases:
        pca = PCA(n_components=2)
        new_coords = pca.fit_transform(flattened_ls)
    else:
        new_coords = np.array(flattened_ls)
    
    xs = new_coords[:, 0]
    ys = new_coords[:, 1]
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs, ys, cross_entropies, s=5)

    if xyzlabel:
        ax.set_xlabel(xyzlabel[0])
        ax.set_ylabel(xyzlabel[1])
        ax.set_zlabel(xyzlabel[2], labelpad=zlabelpad)
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label', labelpad=zlabelpad)
    
    if name:
        if save_to_all:
            plt.savefig(f'imgs/{name}/{name}_3D_error_space_all_weights.pgf')
        else:
            plt.savefig(f'imgs/{name}/{name}_3D_error_space_predefined_weights.pgf')
    

def visualize_3D_err_space(X, y, weight_combs, act_function='relu', biases=True, indexes_to_keep=None, name=None, save_to_all=False, dont_show=False, xyzlabel=None, zlabelpad=None):
    cross_entropies = []
    flattened_ls = []
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
        cross_entropies.append(compute_single_err(X, y, comb, act_function=act_function))
    
    if indexes_to_keep:
        new_coords = np.array(flattened_ls)
        new_coords = new_coords[:, indexes_to_keep]
    elif biases:
        pca = PCA(n_components=2)
        new_coords = pca.fit_transform(flattened_ls)
    else:
        new_coords = np.array(flattened_ls)
    
    xs = new_coords[:, 0]
    ys = new_coords[:, 1]
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
#     ax.scatter(xs, ys, cross_entropies, s=5)
    ax.plot_trisurf(xs, ys, cross_entropies, linewidth=0.2, antialiased=True)

    if xyzlabel:
        ax.set_xlabel(xyzlabel[0])
        ax.set_ylabel(xyzlabel[1])
        ax.set_zlabel(xyzlabel[2], labelpad=zlabelpad)
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label', labelpad=zlabelpad)
    
    if name:
        if save_to_all:
            plt.savefig(f'imgs/{name}/{name}_3D_error_space_all_weights.pgf')
        else:
            plt.savefig(f'imgs/{name}/{name}_3D_error_space_predefined_weights.pgf')

#     if not dont_show:
#         plt.show()
    
    
def visualize_weights3D(weight_combs, name):
    flattened_ls = []
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
    
    flattened_ls = np.array(flattened_ls)
    xs = flattened_ls[:, 0]
    ys = flattened_ls[:, 1]
    zs = flattened_ls[:, 2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs, ys, zs, s=5)
    ax.set_aspect('auto')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_zlabel('bias')
    
    print(flattened_ls)
    
#     plt.savefig(f'imgs/{name}/{name}_converged_weights3D', dpi=300)
    plt.savefig(f'imgs/{name}/{name}_converged_weights3D.pgf')

    
    
def visualize_weights2D(weight_combs, name):
    flattened_ls = []
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
    
    flattened_ls = np.array(flattened_ls)
    xs = flattened_ls[:, 0]
    ys = flattened_ls[:, 1]
    zs = flattened_ls[:, 2]
    
    fig, ax = plt.subplots()
    plt.grid(True, which='both')

    ax.scatter(xs, ys, s=5)
    ax.set_aspect('equal')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    
#     plt.savefig(f'imgs/{name}/{name}_converged_weights2D_without_bias', dpi=300)
    plt.savefig(f'imgs/{name}/{name}_converged_weights2D_without_bias.pgf')
    
