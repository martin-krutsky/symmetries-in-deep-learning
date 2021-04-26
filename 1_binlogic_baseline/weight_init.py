import numpy as np

def init_weight_grid(dim_list, start, end, num, biases=True, sym_weights=None):
    total_nr = 0
    shapes = []
    for l in range(1, len(dim_list)):
        total_nr += dim_list[l]*(dim_list[l-1] + int(biases))  # +1 for bias
        if sym_weights:
            shapes.append((dim_list[l], (dim_list[l-1] + int(biases))))  # +1 for bias
        else:
            shapes.append((dim_list[l], (dim_list[l-1] + int(biases))))  # +1 for bias
            
    linspaces = [np.linspace(start, end, num) for _ in range(total_nr)]
    mesh_grid = np.array(np.meshgrid(*linspaces)).T.reshape(-1, total_nr)
    
    final_grid = []
    for row in mesh_grid:
        final_grid.append([])
        last_i = 0
        for i, shape in enumerate(shapes):
            nr_needed = shape[0]*shape[1]
            to_append = row[last_i:last_i+nr_needed].reshape(*shape)
            
            if sym_weights:
                to_append_ls = []
                for j, neuron_dict in enumerate(sym_weights[i]):
                    to_append_changed = to_append[j].copy()
                    for where_i, from_i in neuron_dict.items():
                        to_append_changed = np.insert(to_append_changed, where_i, to_append[j, from_i])
                    to_append_ls.append(to_append_changed)
                to_append = np.array(to_append_ls)
            
            if biases:
                final_grid[-1].append(to_append)
            else:
                final_grid[-1].append(np.hstack((to_append, np.zeros((shape[0], 1)))))
            last_i += nr_needed

    return final_grid
