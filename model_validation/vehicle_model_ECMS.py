import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
# from models.cell_model import CellModel
from cell_model import CellModel
from scipy import interpolate
import pickle


class Environment_ECMS:
    def __init__(self, cell_model, cycle_path, battery_path, motor_path):
        self.cell_model = cell_model

        self.version = 1

        self.vehicle_comp = {
            "m_veh": 1200,
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
            "cell_number": 400,
            "effective_area_cell": 200.0,
            "max_current_density": 1.0,
            "idling_current_density": 0.00001,
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
        # drv_cycle = sio.loadmat(cycle_path)
        # self.v_veh = drv_cycle["sch_cycle"][:, 1]
        self.v_veh = np.array(cycle_path)
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

        self.action_grid = np.linspace(self.stack_comp["idling_current_density"],
                                       self.stack_comp["max_current_density"],
                                       20)

    def PMP_calculation(self, EF):
        self.SOC = 0.6
        history = {
            "SOC_traj": [],
            "fc_traj": [],
            "action_traj": [],
            "H_traj": []
        }

        for i in range(len(self.v_veh)):
            action, del_SOC, del_fc, done = self.step(i, EF)

            if done:
                break

            self.SOC = min(self.SOC + del_SOC, 1)

            # print("del SOC : {:.3f}, SOC: {:.3f}".format(del_SOC, self.SOC))
            history["SOC_traj"].append(self.SOC)
            history["fc_traj"].append(del_fc)
            history["action_traj"].append(action)

        return history

    def step(self, step, EF):
        done = True
        H, action, del_SOC, del_fc = None, None, None, None

        tq_mot = self.tq_out[step]
        sp_mot = self.sp_out[step]
        p_mot = self.p_mot[step]
        con_mot = self.condition_check_motor(sp_mot, tq_mot)
        if con_mot > 0:
            print("Constraint error, motor cannot follow power")
        else:
            cell_voltages = np.array([self.cell_model.get_voltage(j_fc) for j_fc in self.action_grid])
            stack_voltages = self.stack_comp["cell_number"] * cell_voltages
            stack_currents = self.stack_comp["effective_area_cell"] * self.action_grid
            stack_powers = stack_voltages * stack_currents
            battery_powers = p_mot - stack_powers

            Hs, del_SOCs, del_fuels = self.get_vectors_with_pbat(battery_powers, stack_currents, EF)
            if np.sum(np.isnan(Hs)) == len(Hs):
                print("There is invalid actions..")
            else:
                done = False

                Hs[np.isnan(Hs)] = 10 ** 6
                H = np.min(Hs)
                action = np.argmin(Hs)
                del_SOC = del_SOCs[action]
                del_fc = del_fuels[action]

            if self.SOC < 0.005:
                print("SOC is too low..")
                done = True

        return action, del_SOC, del_fc, done

    def get_vectors_with_pbat(self, battery_powers, stack_currents, EF):
        del_SOC_with_action = self.get_del_soc(battery_powers)
        del_fuel_with_action = self.cal_fuel_consumption(stack_currents)
        H_with_action = del_fuel_with_action + EF * del_SOC_with_action
        return H_with_action, del_SOC_with_action, del_fuel_with_action

    def get_del_soc(self, battery_powers):
        v_dis, v_cha, r_dis, r_cha, i_lim_dis, i_lim_cha = self.get_battery_state()
        del_is = (1 / (2 * r_cha)) * (v_cha - (v_cha ** 2 - 4 * r_cha * battery_powers) ** (0.5)) * (battery_powers < 0) \
                + (1 / (2 * r_dis)) * (v_dis - (v_dis ** 2 - 4 * r_dis * battery_powers) ** (0.5)) * (battery_powers >= 0)
        del_SOCs = - del_is * (self.calculation_comp["del_t"] / self.battery['Q_cap'])
        del_SOCs[((i_lim_dis - del_is) * (i_lim_cha - del_is)) > 0] = np.nan
        return del_SOCs

    def cal_fuel_consumption(self, stack_currents):
        hydrogen_excess_ratio = 1.0
        fuel_consumptions = self.stack_comp["cell_number"] * self.stack_comp["molar_mass_H2"] \
                           / (2 * self.stack_comp["Faraday_constant"]) * stack_currents * hydrogen_excess_ratio
        return fuel_consumptions

    def condition_check_motor(self, sp_mot, tq_mot):
        # con
        con_mot = (tq_mot > np.interp(sp_mot, self.motor["sp_full"], self.motor["tq_max"])) + \
            + (tq_mot < np.interp(sp_mot, self.motor["sp_full"], self.motor["tq_min"])) + \
            + (sp_mot > self.motor["sp_max"]) + (sp_mot < self.motor["sp_min"])
        return con_mot

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


# driving_cycle_path = '../../OC_SIM_DB/OC_SIM_DB_Cycles/Highway/01_FTP72_fuds.mat'
# battery_path = "../../OC_SIM_DB/OC_SIM_DB_Bat/OC_SIM_DB_Bat_nimh_6_240_panasonic_MY01_Prius.mat"
# motor_path = "../../OC_SIM_DB/OC_SIM_DB_Mot/OC_SIM_DB_Mot_pm_95_145_X2.mat"
# cell_model = CellModel()
#
# env = Environment(cell_model, driving_cycle_path, battery_path, motor_path, 10)
# history = env.PMP_calculation()
#
# x = 0
