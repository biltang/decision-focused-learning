def shortest_path_solver(costs, size, sens = 1e-4):
    # Forward Pass
    starting_ind = 0
    starting_ind_c = 0
    samples = costs.shape[0]
    V_arr = torch.zeros(samples, size ** 2)
    for i in range(0, 2 * (size - 1)):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i + 2, 9 - i - 1)
        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        V_1 = V_arr[:, starting_ind:starting_ind + num_nodes]
        layer_costs = costs[:, starting_ind_c:starting_ind_c + num_arcs]
        l_costs = layer_costs[:, 0::2]
        r_costs = layer_costs[:, 1::2]
        next_V_val_l = torch.ones(samples, num_nodes_next) * float('inf')
        next_V_val_r = torch.ones(samples, num_nodes_next) * float('inf')
        if num_nodes_next > num_nodes:
            next_V_val_l[:, :num_nodes_next - 1] = V_1 + l_costs
            next_V_val_r[:, 1:num_nodes_next] = V_1 + r_costs
        else:
            next_V_val_l = V_1[:, :num_nodes_next] + l_costs
            next_V_val_r = V_1[:, 1:num_nodes_next + 1] + r_costs
        next_V_val = torch.minimum(next_V_val_l, next_V_val_r)
        V_arr[:, starting_ind + num_nodes:starting_ind + num_nodes + num_nodes_next] = next_V_val

        starting_ind += num_nodes
        starting_ind_c += num_arcs

    # Backward Pass
    starting_ind = size ** 2
    starting_ind_c = costs.shape[1]
    prev_act = torch.ones(samples, 1)
    sol = torch.zeros(costs.shape)
    for i in range(2 * (size - 1), 0, -1):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i, 9 - i + 1)
        V_1 = V_arr[:, starting_ind - num_nodes:starting_ind]
        V_2 = V_arr[:, starting_ind - num_nodes - num_nodes_next:starting_ind - num_nodes]

        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        layer_costs = costs[:, starting_ind_c - num_arcs: starting_ind_c]

        if num_nodes < num_nodes_next:
            l_cs_res = ((V_2[:, :num_nodes_next - 1] - V_1 + layer_costs[:, ::2]) < sens) * prev_act
            r_cs_res = ((V_2[:, 1:num_nodes_next] - V_1 + layer_costs[:, 1::2]) < sens) * prev_act
            prev_act = torch.zeros(V_2.shape)
            prev_act[:, :num_nodes_next - 1] += l_cs_res
            prev_act[:, 1:num_nodes_next] += r_cs_res
        else:
            l_cs_res = ((V_2 - V_1[:, :num_nodes - 1] + layer_costs[:, ::2]) < sens) * prev_act[:, :num_nodes - 1]
            r_cs_res = ((V_2 - V_1[:, 1:num_nodes] + layer_costs[:, 1::2]) < sens) * prev_act[:, 1:num_nodes]
            prev_act = torch.zeros(V_2.shape)
            prev_act += l_cs_res
            prev_act += r_cs_res
        cs = torch.zeros(layer_costs.shape)
        cs[:, ::2] = l_cs_res
        cs[:, 1::2] = r_cs_res
        sol[:, starting_ind_c - num_arcs: starting_ind_c] = cs

        starting_ind = starting_ind - num_nodes
        starting_ind_c = starting_ind_c - num_arcs
    # Dimension (samples, num edges)
    obj = torch.sum(sol * costs, axis=1)
    # Dimension (samples, 1)
    return sol, obj.reshape(-1,1)

# optimization model
class optGenModel(optModel):
    """
    This is an abstract class for Pyomo-based optimization model

    Attributes:
        _model (PyOmo model): Pyomo model
        solver (str): optimization solver in the background
    """

    def __init__(self):
        """
        Args:
            solver (str): optimization solver in the background
        """
        super().__init__()
        # init obj
        if self._model.modelSense == EPO.MINIMIZE:
            self.modelSense = EPO.MINIMIZE
        if self._model.modelSense == EPO.MAXIMIZE:
            self.modelSense = EPO.MAXIMIZE

    def __repr__(self):
        return "optGenModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        self._model.costvec = c

    def copy(self):
        """
        A method to copy model

        Returns:
            optModel: new copied model
        """
        new_model = copy(self)
        return new_model

    def addConstr(self):
        new_model = self.copy()
        # add constraint
        return new_model

class modelclass():
    def __init__(self, size):
        self.size = size
        self.costvec = None
        self.modelSense = EPO.MINIMIZE
        self.x = np.ones(2 * size * (size - 1))
class shortestPathModel(optGenModel):

    def __init__(self):
        self.grid = (5,5)
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        m = modelclass(self.grid[0])
        x = m.x
        # sense
        m.modelSense = EPO.MINIMIZE
        return m, x

    def solve(self):
        sol, obj = shortest_path_solver(self._model.costvec.reshape(-1,len(self.x)), self._model.size)
        return sol, obj