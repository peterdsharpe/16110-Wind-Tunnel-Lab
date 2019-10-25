### process_data.py
# Takes the raw data and processes it to get useful information

import numpy as np
import torch
import get_raw_data as raw_data
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

# Set up Pint
u = pint.UnitRegistry()

# Make a function to get the standard deviation of a mean
def get_stdev_of_mean(data):
    """
    Finds the standard deviation of the mean of a dataset
    :param data: 1D ndarray
    :return:
    """
    n_samples = len(data)
    stdev_of_data = np.std(data)

    stdev_of_mean = 1 / np.sqrt(n_samples) * stdev_of_data

    return stdev_of_mean


### Determine the load cell slope and offset
# First, get mean values
mean_load_cell_voltage_no_weight_before_experiment = torch.tensor(
    np.mean(raw_data.wt_load_cell_tare_no_weight_before_experiments[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_no_weight_after_experiment = torch.tensor(
    np.mean(raw_data.wt_load_cell_tare_no_weight_after_experiments[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_2oz_after_experiment = torch.tensor(
    np.mean(raw_data.wt_load_cell_tare_2oz_weight_after_experiments[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_8oz_before_experiment = torch.tensor(
    np.mean(raw_data.wt_load_cell_tare_8oz_weight_before_experiments[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_8oz_after_experiment = torch.tensor(
    np.mean(raw_data.wt_load_cell_tare_8oz_weight_after_experiments[:, 0]),
    requires_grad=True
)

# And standard deviations.
stdev_load_cell_voltage_no_weight_before_experiment = torch.tensor(
    get_stdev_of_mean(raw_data.wt_load_cell_tare_no_weight_before_experiments[:, 0])
)
stdev_load_cell_voltage_no_weight_after_experiment = torch.tensor(
    get_stdev_of_mean(raw_data.wt_load_cell_tare_no_weight_after_experiments[:, 0])
)
stdev_load_cell_voltage_2oz_after_experiment = torch.tensor(
    get_stdev_of_mean(raw_data.wt_load_cell_tare_2oz_weight_after_experiments[:, 0])
)
stdev_load_cell_voltage_8oz_before_experiment = torch.tensor(
    get_stdev_of_mean(raw_data.wt_load_cell_tare_8oz_weight_before_experiments[:, 0])
)
stdev_load_cell_voltage_8oz_after_experiment = torch.tensor(
    get_stdev_of_mean(raw_data.wt_load_cell_tare_8oz_weight_after_experiments[:, 0])
)

# What are the weights of the calibration pieces?
calibration_mass_oz = torch.tensor((
    0, 0, 2, 8, 8
))

newtons_per_oz_force = (1 * u.oz * (9.81 * u.m / u.s ** 2)).to('N').magnitude
calibration_force_N = calibration_mass_oz * newtons_per_oz_force

# What are the calibration_voltages?
calibration_voltages = torch.stack([
    mean_load_cell_voltage_no_weight_before_experiment,
    mean_load_cell_voltage_no_weight_after_experiment,
    mean_load_cell_voltage_2oz_after_experiment,
    mean_load_cell_voltage_8oz_before_experiment,
    mean_load_cell_voltage_8oz_after_experiment
])

# Do the regression
[load_cell_slope, load_cell_offset] = np.polyfit(
    calibration_voltages.detach().numpy(),
    calibration_force_N.detach().numpy(),
    deg=1
)

# Plot the regression
style.use("seaborn")
plt.scatter(calibration_voltages.detach().numpy(), calibration_force_N.detach().numpy(), label="Calibration Data")
calibration_voltages_for_plotting = torch.linspace(
    calibration_voltages.detach().min(),
    calibration_voltages.detach().max(),
    100
)
calibration_forces_for_plotting = load_cell_slope * calibration_voltages_for_plotting + load_cell_offset
plt.plot(calibration_voltages_for_plotting, calibration_forces_for_plotting, 'g', label="Linear Fit")
plt.title("Load Cell Linearity")
plt.xlabel("Voltage [V]")
plt.ylabel("Force [N]")
plt.legend()
plt.show()

### Use the regression to find the load at each test case
# Find the voltages at each speed
mean_load_cell_voltage_approx_30_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_30_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_35_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_35_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_40_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_40_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_50_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_50_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_60_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_60_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_70_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_70_mph[:, 0]),
    requires_grad=True
)
mean_load_cell_voltage_approx_73_mph = torch.tensor(
    np.mean(raw_data.wt_raw_data_73_mph[:, 0]),
    requires_grad=True
)

voltages_at_speed = torch.stack((
    mean_load_cell_voltage_approx_30_mph,
    mean_load_cell_voltage_approx_35_mph,
    mean_load_cell_voltage_approx_40_mph,
    mean_load_cell_voltage_approx_50_mph,
    mean_load_cell_voltage_approx_60_mph,
    mean_load_cell_voltage_approx_70_mph,
    mean_load_cell_voltage_approx_73_mph
))

# Find the forces at each speed
forces_at_speed_in_newtons = (load_cell_slope * voltages_at_speed + load_cell_offset)

### Find the dynamic pressure at each speed
# Find the raw dynamic pressure at each speed (Units of torr)
mean_raw_q_approx_30_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_30_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_35_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_35_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_40_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_40_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_50_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_50_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_60_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_60_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_70_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_70_mph[:, 1]),
    requires_grad=True
)
mean_raw_q_approx_73_mph_in_torr = torch.tensor(
    np.mean(raw_data.wt_raw_data_73_mph[:, 1]),
    requires_grad=True
)

# Find the standard deviations of dynamic pressure
stdev_raw_q_approx_30_mph_in_torr = torch.tensor(
    get_stdev_of_mean(raw_data.wt_raw_data_30_mph[:, 1])
)

raw_q_at_speed_in_torr = torch.stack((
    mean_raw_q_approx_30_mph_in_torr,
    mean_raw_q_approx_35_mph_in_torr,
    mean_raw_q_approx_40_mph_in_torr,
    mean_raw_q_approx_50_mph_in_torr,
    mean_raw_q_approx_60_mph_in_torr,
    mean_raw_q_approx_70_mph_in_torr,
    mean_raw_q_approx_73_mph_in_torr
))

raw_q_at_zero_speed_in_torr = torch.tensor(
    np.mean(raw_data.wt_q_tare_before_experiments[:, 1])
)

q_at_speed_in_torr = raw_q_at_speed_in_torr - raw_q_at_zero_speed_in_torr

pascals_per_torr = (1 * u.torr).to("Pa").magnitude

q_at_speed_in_Pa = q_at_speed_in_torr * pascals_per_torr
