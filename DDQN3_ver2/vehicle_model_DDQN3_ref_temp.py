import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
# from models.cell_model import CellModel
from cell_model import CellModel
from scipy import interpolate
import pickle


class Environment:
    def __init__(self, cell_model, cycle_path, battery_path, motor_path, reward_factor):
        self.cell_model = cell_model
        self.reward_factor = reward_factor

        self.version = 2

        self.vehicle_comp = {
            "m_veh": 900,
            "cd": 0.456,
            "fa": 2.4,
            "rr": 0.009,
            "radius_tire": 0.337,
            "r_final": 3.648,
            "eff_gear": 0.97,
            "rho_air": 1.24,
            "g": 9.8,
        }
        self.stack_comp = {
            "cell_number": 200,
            "effective_area_cell": 200.0,
            "max_current_density": 1.0,
            "idling_current_density": 0.01,
            "Faraday_constant": 96485,
            "molar_mass_H2": 2.016,
        }
        self.calculation_comp = {
            "del_t": 1,
            "j_resolution": 10,
            "action_size": 20,
            "state_size": 3,
        }

        ####  Extract Cycle Information  #####
        drv_cycle = sio.loadmat(cycle_path)
        self.v_veh = drv_cycle["sch_cycle"][:, 1]
        self.total_distance = np.sum(self.v_veh)
        self.v_grade = np.zeros(self.v_veh.shape)
        self.a_veh = (np.append(self.v_veh[1:], self.v_veh[-1]) - np.append(self.v_veh[0], self.v_veh[:-1])) \
                     / (2 * self.calculation_comp["del_t"])

        resistance_inertia = self.vehicle_comp["m_veh"] * self.a_veh
        resistance_friction = self.vehicle_comp["m_veh"] * self.vehicle_comp["g"] * np.cos(self.v_grade * np.pi / 180) \
        * self.vehicle_comp["rr"] * (self.v_veh > 0)
        resistance_climbing = self.vehicle_comp["m_veh"] * self.vehicle_comp["g"] * np.sin(self.v_grade * np.pi / 180) \
        * (self.v_veh > 0)
        resistance_air = 0.5 * self.vehicle_comp["cd"] * self.vehicle_comp["fa"] * self.vehicle_comp["rho_air"] * \
            self.v_veh ** 2

        self.sp_wheel = self.v_veh / self.vehicle_comp["radius_tire"]
        self.tq_wheel = self.vehicle_comp["radius_tire"] * (resistance_inertia + resistance_friction +
                                                            resistance_climbing + resistance_air)

        self.sp_out = self.sp_wheel * self.vehicle_comp["r_final"]
        self.tq_out = (self.tq_wheel / self.vehicle_comp["eff_gear"] * (self.tq_wheel >= 0) + self.tq_wheel \
                      * self.vehicle_comp["eff_gear"] * (self.tq_wheel < 0)) / self.vehicle_comp["r_final"]
        self.power_out = self.sp_out * self.tq_out
        self.power_out_norm = (self.power_out - self.power_out.mean()) / self.power_out.std()


        ####  Extract Motor Information  #####
        motor_comp = sio.loadmat(motor_path)["Mot"]
        self.motor = {
            "sp": motor_comp["sp"][0][0][0],
            "tq": motor_comp["tq"][0][0][0],
            "tq_max": motor_comp["tq_max"][0][0][0],
            "tq_min": motor_comp["tq_min"][0][0][0],
            "eff": motor_comp["eff"][0][0],
            "sp_full": motor_comp["sp_full"][0][0][0]
        }
        self.motor["tq_max"] = np.minimum(np.max(self.motor["tq"]), self.motor["tq_max"])
        self.motor["tq_min"] = np.maximum(np.min(self.motor["tq"]), self.motor["tq_min"])
        self.motor["sp_max"] = np.max(self.motor["sp"])
        self.motor["sp_min"] = np.min(self.motor["sp"])
        self.motor["eff_map"] = interpolate.interp2d(self.motor["sp"], self.motor["tq"], self.motor["eff"])
        self.tq_mot = self.tq_out
        self.sp_mot = self.sp_out
        self.p_mot = np.array([(self.motor["eff_map"](sp_mot, tq_mot) * sp_mot * tq_mot)[0] for sp_mot, tq_mot in
                               zip(self.sp_mot, self.tq_mot)])

        ####  Extract Battery Information  #####
        battery_comp = sio.loadmat(battery_path)["Bat"]
        self.battery = {
            "SOC_ind":  battery_comp["SOC_ind"][0][0][0],
            "Vol_dis": battery_comp["Vol_dis"][0][0][0],
            "Vol_cha": battery_comp["Vol_cha"][0][0][0],
            "Res_dis": battery_comp["Res_dis"][0][0][0],
            "Res_cha": battery_comp["Res_cha"][0][0][0],
            "Cur_lim_dis": battery_comp["Cur_lim_dis"][0][0][0],
            "Cur_lim_cha": battery_comp["Cur_lim_cha"][0][0][0],
            "Q_cap": battery_comp["Q_cap"][0][0][0][0],
        }

        self.step_num = 0
        self.SOC = 0.6
        self.fuel_consumption = 0
        self.cycle_length = len(self.sp_out)
        self.idling_voltage = self.cell_model.get_voltage(self.stack_comp["idling_current_density"])
        self.history = {
            "SOC": [],
            "Action": [],
            "P_stack": [],
            "P_battery": [],
            "P_motor": [],
            "m_fuel": [],
            "j_min": [],
            "j_max": []
        }

    def reset(self):
        self.step_num = 0
        self.SOC = 0.6
        self.fuel_consumption = 0
        # state = [self.tq_out[self.step_num], self.sp_out[self.step_num], self.SOC]
        j_min, j_max, _ = self.get_curdensity_region(self.p_mot[self.step_num])
        state = [self.power_out[self.step_num] / 1000, self.SOC, j_min, j_max]
        # state = [self.power_out_norm[self.step_num], self.SOC - 0.6]
        self.history = {
            "SOC": [],
            "Action": [],
            "P_stack": [],
            "P_battery": [],
            "P_motor": [],
            "m_fuel": [],
            "j_min": [],
            "j_max": []
        }
        return state

    def step(self, action):
        state = None
        reward = None
        done = False

        tq_mot = self.tq_out[self.step_num]
        sp_mot = self.sp_out[self.step_num]
        p_mot = self.p_mot[self.step_num]
        con_mot = self.condition_check_motor(sp_mot, tq_mot)
        if con_mot > 0:
            if p_mot >= 0:
                print("Constraint error, motor cannot follow traction power")
                done = True
            else:
                print("Constraint error, motor cannot follow generator power")

        else:
            j_min, j_max, done = self.get_curdensity_region(p_mot)
            j_fc = np.linspace(j_min, j_max, self.calculation_comp['action_size'])[action]
            cell_voltage = self.cell_model.get_voltage(j_fc)
            stack_voltage = self.stack_comp["cell_number"] * cell_voltage
            stack_current = self.stack_comp["effective_area_cell"] * j_fc
            p_stack = stack_voltage * stack_current
            p_bat = p_mot - p_stack
            self.update_soc(p_bat)
            m_fuel = self.cal_fuel_consumption(stack_current)

            # reward = self.cal_reward(m_fuel)
            # reward = self.cal_reward(m_fuel)
            self.fuel_consumption += m_fuel
            state, reward, done = self.post_process(action, p_stack, p_bat, p_mot, m_fuel, j_min, j_max)

            if np.isnan(self.SOC):
                done = True
                reward = -10000
                print("SOC is nan...")

        return state, reward, done

    def cal_reward(self, m_fuel):
        reward = - m_fuel - self.reward_factor * abs(self.SOC - 0.6)
        return reward

    def cal_reward_2(self, m_fuel):
        distance_ratio = np.sum(self.v_veh[:self.step_num]) / self.total_distance
        reward = self.reward_factor * (distance_ratio > 0.8) * abs(self.SOC - 0.6) + m_fuel
        # print(reward)
        return -reward

    def post_process(self, action, p_stack, p_bat, p_mot, m_fuel, j_min, j_max):
        state = None
        done = False
        reward = self.cal_reward(m_fuel)

        self.history["SOC"].append(self.SOC)
        self.history["Action"].append(action)
        self.history["P_stack"].append(p_stack)
        self.history["P_battery"].append(p_bat)
        self.history["P_motor"].append(p_mot)
        self.history["m_fuel"].append(m_fuel)
        self.history["j_min"].append(j_min)
        self.history["j_max"].append(j_max)

        self.step_num += 1
        if self.step_num == self.cycle_length:
            done = True
            print("maximum steps, simulation is done ... ")
        else:
            j_min, j_max, done = self.get_curdensity_region(self.p_mot[self.step_num])
            state = [self.power_out[self.step_num] / 1000, self.SOC, j_min, j_max]
        return state, reward, done

    def get_curdensity_region(self, p_mot):
        done = False

        j_fc_set = np.linspace(self.stack_comp["idling_current_density"], self.stack_comp["max_current_density"],
                            self.calculation_comp["j_resolution"])
        V_fc_set = np.array([self.cell_model.get_voltage(j_fc) for j_fc in j_fc_set])
        V_stack_set = V_fc_set * self.stack_comp["cell_number"]
        I_stack_set = j_fc_set * self.stack_comp["effective_area_cell"]
        P_stack_set = V_stack_set * I_stack_set

        P_battery_set = p_mot - P_stack_set
        condition_set = [self.condition_check_battery(p_bat) for p_bat in P_battery_set]
        if sum(condition_set) == 0:
            done = True
            j_fc_min, j_fc_max = None, None
            # print(p_mot)
            print("Available condition is not avail... SOC: {}".format(self.SOC))

        else:
            j_fc_set_avail = j_fc_set[condition_set]
            j_fc_min, j_fc_max = j_fc_set_avail[0], j_fc_set_avail[-1]
        return j_fc_min, j_fc_max, done

    def condition_check_motor(self, sp_mot, tq_mot):
        # con
        con_mot = (tq_mot > np.interp(sp_mot, self.motor["sp_full"], self.motor["tq_max"])) + \
            + (tq_mot < np.interp(sp_mot, self.motor["sp_full"], self.motor["tq_min"])) + \
            + (sp_mot > self.motor["sp_max"]) + (sp_mot < self.motor["sp_min"])
        return con_mot

    def condition_check_battery(self, p_bat):
        v_dis, v_cha, r_dis, r_cha, i_lim_dis, i_lim_cha = self.get_battery_state()
        del_i = (1 / (2 * r_cha)) * (v_cha - (v_cha ** 2 - 4 * r_cha * p_bat) ** (0.5)) * (p_bat < 0) + (1 / (
                2 * r_dis)) * (v_dis - (v_dis ** 2 - 4 * r_dis * p_bat) ** (0.5)) * (p_bat >= 0)
        if ((i_lim_dis - del_i) * (i_lim_cha - del_i)) > 0:
            condition = False
        else:
            condition = True
        return condition

    def get_battery_state(self):
        v_dis = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Vol_dis'], assume_sorted=False,
                                     fill_value='extrapolate')(self.SOC)
        v_cha = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Vol_cha'], assume_sorted=False,
                                     fill_value='extrapolate')(self.SOC)
        r_dis = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Res_dis'], assume_sorted=False,
                                     fill_value='extrapolate')(self.SOC)
        r_cha = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Res_cha'], assume_sorted=False,
                                     fill_value='extrapolate')(self.SOC)
        i_lim_dis = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Cur_lim_dis'], assume_sorted=False,
                                         fill_value='extrapolate')(self.SOC)
        i_lim_cha = interpolate.interp1d(self.battery['SOC_ind'], self.battery['Cur_lim_cha'], assume_sorted=False,
                                         fill_value='extrapolate')(self.SOC)
        return [v_dis, v_cha, r_dis, r_cha, i_lim_dis, i_lim_cha]

    def update_soc(self, p_bat):
        v_dis, v_cha, r_dis, r_cha, i_lim_dis, i_lim_cha = self.get_battery_state()

        del_i = (1 / (2 * r_cha)) * (v_cha - (v_cha ** 2 - 4 * r_cha * p_bat) ** (0.5)) * (p_bat < 0) + (1 / (
                2 * r_dis)) * (v_dis - (v_dis ** 2 - 4 * r_dis * p_bat) ** (0.5)) * (p_bat >= 0)

        # if ((i_lim_dis - del_i) * (i_lim_cha - del_i)) > 0:
        #     if del_i > 0:
        #         print("Constraint error, battery current limit ( motor mode )")
        #     else:
        #         print("Constraint error, battery current limit ( generator mode)")
        # else:
        del_soc = -del_i * (self.calculation_comp["del_t"] / self.battery['Q_cap'])
        self.SOC = min(self.SOC + del_soc, 1)

    def cal_fuel_consumption(self, stack_current):
        hydrogen_excess_ratio = 1.0
        fuel_consumption = self.stack_comp["cell_number"] * self.stack_comp["molar_mass_H2"] \
                           / (2 * self.stack_comp["Faraday_constant"]) * stack_current * hydrogen_excess_ratio
        return fuel_consumption


# drving_cycle = './OC_SIM_DB/OC_SIM_DB_Cycles/Highway/01_FTP72_fuds.mat'
# battery_path = "./OC_SIM_DB/OC_SIM_DB_Bat/OC_SIM_DB_Bat_e-4wd_Battery.mat"
# motor_path = "./OC_SIM_DB/OC_SIM_DB_Mot/OC_SIM_DB_Mot_id_75_110_Westinghouse.mat"
# cell_model = CellModel()
# env = Environment(cell_model, drving_cycle, battery_path, motor_path, 0.1)
#
# for i in range(env.cycle_length):
#     print(i)
#     action = np.random.choice(10, 1)[0]
#     state, reward, done = env.step(action)
#     if done:
#         history = env.history
#         with open("./history.p", "wb") as fp:
#             pickle.dump(history, fp, protocol=pickle.HIGHEST_PROTOCOL)
#         break



