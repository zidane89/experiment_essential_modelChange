import numpy as np
import scipy.io as sio
import glob
import matplotlib.pyplot as plt

class Driver_MDP:
    def __init__(self, eps):
        self.driving_cycle_paths = glob.glob("../data/driving_cycles/all/*.mat")

        self.v_max, self.a_min, self.a_max = self.get_min_max()
        self.a_min = - self.a_max

        self.v_grid_num = 20
        self.a_grid_num = 20
        self.v_grid = np.linspace(0, self.v_max, self.v_grid_num + 1)
        self.a_grid = np.linspace(self.a_min, self.a_max, self.a_grid_num + 1)

        self.transition_matrix, self.transition_prob_matrix = self.get_transition_matrix(eps)


    def get_min_max(self):
        v_max = 0
        v_diff_min = 0
        v_diff_max = 0

        for cycle_path in self.driving_cycle_paths:
            drv_cycle = sio.loadmat(cycle_path)
            v_veh = drv_cycle["sch_cycle"][:, 1]
            v_diff_veh = np.diff(v_veh)

            for v in v_veh:
                if v > v_max:
                    v_max = v

            for v_diff in v_diff_veh:
                if v_diff > v_diff_max:
                    v_diff_max = v_diff
                if v_diff < v_diff_min:
                    v_diff_min = v_diff
        return v_max, v_diff_min, v_diff_max

    def get_indices(self, v, a):
        i, j = None, None

        for i in range(len(self.v_grid) - 1):
            if self.v_grid[i] <= v < self.v_grid[i + 1]:
                break

        for j in range(len(self.a_grid) - 1):
            if self.a_grid[j] <= a < self.a_grid[j + 1]:
                break
        return i, j

    def get_transition_matrix(self, eps):
        transition_matrix = np.zeros((self.v_grid_num, self.a_grid_num))

        for cycle_path in self.driving_cycle_paths:
            drv_cycle = sio.loadmat(cycle_path)
            v_veh = drv_cycle["sch_cycle"][:, 1]

            for t in range(len(v_veh) - 1):
                v = v_veh[t]
                a = v_veh[t + 1] - v_veh[t]

                i, j = self.get_indices(v, a)
                transition_matrix[i, j] += 1

        transition_prob_matrix = np.zeros_like(transition_matrix)
        # eps = 0.02

        for i in range(len(transition_matrix)):
            transition_matrix[i] += np.sum(transition_matrix[i]) * eps
            state_sum = np.sum(transition_matrix[i])

            state_prob = transition_matrix[i] / state_sum
            transition_prob_matrix[i] = state_prob

        return transition_matrix, transition_prob_matrix

    def mapping_v(self, v_continue):
        i = None
        for i in range(len(self.v_grid) - 1):
            if self.v_grid[i] <= v_continue < self.v_grid[i + 1]:
                break
        v_category = i
        return v_category

    def mapping_a(self, v_category):
        probs = self.transition_prob_matrix[v_category]
        a_category = np.random.choice(len(probs), p=probs)
        a_continue = self.a_grid[a_category]
        return a_continue

    def get_cycle(self):
        cycle_length = 1000

        cycle = [0]
        v_t_con = 0
        for t in range(cycle_length):
            v_t_cat = self.mapping_v(v_t_con)
            a_t_con = self.mapping_a(v_t_cat)

            v_t_con += a_t_con
            v_t_con = max(min(v_t_con, self.v_max), 0)
            cycle.append(v_t_con)
        return cycle


