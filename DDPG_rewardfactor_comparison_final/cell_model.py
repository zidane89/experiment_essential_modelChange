import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


class CellModel:
    def __init__(self):
        self.constants = {
            # "theremo_voltage": 1.0,  # V (variable, complete)
            "temperature": 343,  # A/cm2
            # "vapor_saturation_pressure": 0.307,  # atm  (variable, complete)
            "hydrogen_mole_fraction": 0.9,  # (variable, have to find appropriate value)
            "oxygen_mole_fraction": 0.19,  # (variable, have to find appropriate value)
            "anode_water_mole_fraction": 0.1,  # (variable, have to find appropriate value)
            "cathode_water_mole_fraction": 0.1,  # (variable, have to find appropriate value)
            "cathode_pressure": 3,   # atm
            "anode_pressure": 3,  # atm
            # "effective_hydrogen_diffusivity": 0.149,  # cm2/s  (variable, complete)
            # "effective_oxygen_diffusivity": 0.0295,  # cm2/s  (variable, complete)
            "water_diffusivity_nafion": 3.81e-6,  # cm2/s  (variable, coupled)
            "transfer_coefficient": 0.5,  # (variable, complete)
            "exchange_current_density": 0.0001,  # A/cm2  (variable, can not find EQ)
            "electrolyte_thickness": 125,  # um
            "anode_thickness": 350,  # um
            "cathode_thickness": 350,  # um
            "gas_constant": 8.314,  # J/mol*K
            "Faraday_constant": 96485,  # C/mol
            "limiting_current_density": 2.5  # A/cm2
        }

    def conductivity_integrand(self, x, a, b, c):
        return 1 / (a + b * np.exp(c * x))

    def get_thermo_voltage(self):
        s_0 = -162.76
        T_0 = 300
        E_0 = 1.229
        P_h2 = self.constants["anode_pressure"] * self.constants["hydrogen_mole_fraction"]
        P_o2 = self.constants["cathode_pressure"] * self.constants["oxygen_mole_fraction"]
        n = 2
        F = self.constants["Faraday_constant"]
        R = self.constants["gas_constant"]
        T = self.constants["temperature"]

        E = E_0 + s_0 / (n * F) * (T - T_0) - R * T / (n * F) * np.log(1 / (P_h2 * P_o2 ** 0.5))
        return E

    def get_effective_diffusivity(self):
        a = 3.64 * 10 ** -4
        b = 2.334
        T = self.constants["temperature"]
        porosity = 0.4
        tortuosity = 1.5

        # H2 - H20 in Anode
        P_anode = self.constants["anode_pressure"]
        T_H2 = 33.3
        T_H2O = 647.3
        P_H2 = 12.8
        P_H2O = 217.5
        M_H2 = 2.016
        M_H2O = 18.015

        therm1 = (T / np.sqrt(T_H2 * T_H2O)) ** b
        therm2 = (P_H2 * P_H2O) ** (1 / 3)
        therm3 = (T_H2 * T_H2O) ** (5 / 12)
        therm4 = (1 / M_H2 + 1 / M_H2O) ** (1 / 2)
        D_normal_H2_H20 = a * therm1 * therm2 * therm3 * therm4 / P_anode
        D_eff_H2_H2O = D_normal_H2_H20 * porosity ** tortuosity

        # O2 - H2O in Cathode
        P_cathode = self.constants["cathode_pressure"]
        T_O2 = 154.4
        T_H2O = 647.3
        P_O2 = 49.7
        P_H2O = 217.5
        M_O2 = 31.999
        M_H2O = 18.015

        therm1 = (T / np.sqrt(T_O2 * T_H2O)) ** b
        therm2 = (P_O2 * P_H2O) ** (1 / 3)
        therm3 = (T_O2 * T_H2O) ** (5 / 12)
        therm4 = (1 / M_O2 + 1 / M_H2O) ** (1 / 2)
        D_normal_O2_H20 = a * therm1 * therm2 * therm3 * therm4 / P_cathode
        D_eff_O2_H2O = D_normal_O2_H20 * porosity ** tortuosity
        return D_eff_H2_H2O, D_eff_O2_H2O

    def get_nafion_diffusivity(self):
        pass

    def get_vapor_saturation_pressure(self):
        T = self.constants["temperature"] - 273
        alpha = -2.1794 + 0.02953 * T - 9.1837 * (10 ** -5) * T ** 2 + 1.4454 * (10 ** -7) * T ** 3
        P_sat = 10 ** alpha
        return P_sat

    def get_loss(self, current_density):
        j = current_density
        j_0 = self.constants["exchange_current_density"]
        P_sat = self.get_vapor_saturation_pressure()
        P_a = self.constants["anode_pressure"]
        P_c = self.constants["cathode_pressure"]
        F = self.constants["Faraday_constant"]
        D_1, D_2 = self.get_effective_diffusivity()
        D_lambda = self.constants["water_diffusivity_nafion"]
        t_a = self.constants["anode_thickness"]
        t_m = self.constants["electrolyte_thickness"]
        t_c = self.constants["cathode_thickness"]
        R = self.constants["gas_constant"]
        T = self.constants["temperature"]
        x_a = self.constants["anode_water_mole_fraction"]
        x_c = self.constants["cathode_water_mole_fraction"]
        x_o = self.constants["oxygen_mole_fraction"]
        alpha_transfer = self.constants["transfer_coefficient"]
        n_drag = 2.5
        rho_air = 0.00197  # kg/cm3

        # ohmic loss
        element_1 = 4.4 + 14 * (P_a / P_sat) * (t_a * 10**-6 * R * T * j / 0.0001) / (2 * F * P_a * 101325 * D_1 * 0.0001)
        element_2 = 1
        element_3 = 4.4 - 4 * (P_c / P_sat) * (t_c * 10**-6 * R * T * j / 0.0001) / (2 * F * P_c * 101325 * D_2 * 0.0001)
        element_4 = np.exp((j * n_drag * t_m * 10**-4) / (22 * F * rho_air * D_lambda))

        target_1 = 14 * (P_a / P_sat) * x_a
        target_2 = 10 + 4 * (P_c / P_sat) * x_c + 4 * (P_c / P_sat) * (t_c * 10**-6 * R * T * j / 0.0001) / (2 * F * P_c * 101325 * D_2 * 0.0001)

        A = np.array([[element_1, element_2], [element_3, element_4]])
        A_inv = np.linalg.inv(A)
        b = np.array([target_1, target_2])

        values = np.matmul(A_inv, b)
        alpha = values[0]
        const = values[1]

        a = 0.02285 * alpha * np.exp(1268 * (1 / 303 - 1 / T)) - 0.00326 * np.exp(1268 * (1 / 303 - 1 / T))
        b = 0.005193 * const * np.exp(1268 * (1 / 303 - 1 / T))
        c = 0.000598 * j / (3.81 * 10 ** -6)
        resistance = quad(self.conductivity_integrand, 0, t_m * 10 ** -4, (a, b, c))[0]
        ohmic_loss = resistance * j

        # cathode activation loss
        ratio = j * 10 ** 4 * R * T / (4 * F * P_c * 101325 * D_2 * 10 ** -4)
        log_content = (j * 1 / (j_0 * P_c)) * (1 / (x_o - t_c * 10**-6 * ratio))
        cathode_loss = R * T / (4 * alpha_transfer * F) * np.log(log_content)

        # concentration loss
        j_L = self.constants["limiting_current_density"]
        concentration_loss = (R * T / (2 * F)) * (1 + 1 / alpha_transfer) * np.log(j_L / (j_L - j))

        return ohmic_loss, cathode_loss, concentration_loss

    def get_voltage(self, current_density):
        thermo_voltage = self.get_thermo_voltage()
        # thermo_voltage = 1
        ohmic_loss, cathode_loss, concentration_loss = self.get_loss(current_density)
        real_voltage = thermo_voltage - ohmic_loss - cathode_loss
        return real_voltage


fuel_cell = CellModel()
# print(fuel_cell.get_loss(0.5))
currents = np.linspace(0.005, 2.4999, 100)
voltages = []
powers = []
for cur in currents:
    voltage = fuel_cell.get_voltage(cur)
    voltages.append(voltage)
    powers.append(voltage * cur)

# plt.plot(currents, voltages, linewidth=2.0)
# plt.plot(currents, powers)
# plt.xlabel("Current Density [A/Cm^2]")
# plt.ylabel("Voltage")
# plt.grid()
# plt.show()

# save_dict = {
#     "currents": currents,
#     "voltages": voltages
# }
# with open("results_model_origin.pickle", "wb") as handle:
#     pickle.dump(save_dict, handle)


#
# # fuel_cell.get_voltage(0.5)
# fuel_cell.get_vapor_saturation_pressure()







