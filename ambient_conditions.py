import numpy as np
import torch
import get_raw_data as raw_data
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

u = pint.UnitRegistry()

Pa_per_inHg = (1 * u("inHg")).to("Pa").magnitude

pressure_Pa = torch.tensor(30.28 * Pa_per_inHg, requires_grad=True)

temperature_K = torch.tensor(20.5 + 273.15, requires_grad=True)

# Use equation of state to find density
R = 287 # * u("J/kg/K")

# P = rho*R*T

rho_kgm3 = pressure_Pa/(R*temperature_K) #(pressure_Pa / (R * temperature_K)).to("kg/m^3").magnitude

# Use Sutherland's law to find viscosity
mu_0 = 1.716e-5 # kg/m-s
T0 = 273.15 # K
S = 110.4 # K
# C1 = 1.458e-6 # kg/(m*s*sqrt(K))

mu = mu_0*(temperature_K / T0)**(3/2) * (T0 + S)/(temperature_K + S) # kg/(m*s)