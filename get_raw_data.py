### get_raw_data.py
# Imports all raw data from the *.csv files taken during the experiment.

import numpy as np


def parse_wind_tunnel_csv(filename):
    """
    Parses the CSV files written during the experiment.
    :param filename: Name of the file to be read.
    :return: Data, in the format as follows:
    ### All of the wind tunnel data files are Nx2 arrays in the format [load_cell_voltage, measured_dynamic_pressure_in_torr].
    ### Each row represents a separate data collect (0.001 sec spacing).
    ### Note that the load cell voltage should be converted to load via a linear fit to tare data.
    ### Note that the dynamic pressure measurements should be calibrated against the tare values.
    """
    raw_data = np.genfromtxt(
        filename,
        skip_header=7,
        delimiter=',',
        usecols=(2, 3),
        dtype='str'
    )
    data = np.zeros(raw_data.shape)
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            data[i, j] = float(raw_data[i, j].split("\"")[1])

    return data


### Wind Tunnel (wt) Data Format:
### All of the wind tunnel data files are Nx2 arrays in the format [load_cell_voltage, measured_dynamic_pressure_in_torr].
### Each row represents a separate data collect (0.001 sec spacing).
### Note that the load cell voltage should be converted to load via a linear fit to tare data.
### Note that the dynamic pressure measurements should be calibrated against the tare values.

# Load cell tares before experiments
wt_load_cell_tare_no_weight_before_experiments = parse_wind_tunnel_csv(
    'data/load_cell_tare_no_weight_before_all_experiments/Analog - 10-24-2019 10-29-09.787 AM.csv')
wt_load_cell_tare_8oz_weight_before_experiments = parse_wind_tunnel_csv(
    'data/load_cell_tare_8oz_weight_before_all_experiments/Analog - 10-24-2019 10-31-00.065 AM.csv')

# Dynamic pressure tare before experiment (this is redundant from prior data, probably won't use this)
wt_q_tare_before_experiments = parse_wind_tunnel_csv(
    'data/load_cell_and_q_tare_no_weight_no_speed_before_all_experiments/Analog - 10-24-2019 10-36-31.486 AM.csv')

# Airfoil at-speed data
wt_raw_data_30_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_30_mph/Analog - 10-24-2019 10-40-07.697 AM.csv')
wt_raw_data_35_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_35_mph/Analog - 10-24-2019 10-43-23.611 AM.csv')
wt_raw_data_40_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_40_mph/Analog - 10-24-2019 10-45-19.445 AM.csv')
wt_raw_data_50_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_50_mph/Analog - 10-24-2019 10-47-22.874 AM.csv')
wt_raw_data_60_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_60_mph/Analog - 10-24-2019 10-49-37.397 AM.csv')
wt_raw_data_70_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_70_mph/Analog - 10-24-2019 10-54-41.000 AM.csv')
wt_raw_data_73_mph = parse_wind_tunnel_csv(
    'data/transition_measurement_approx_73_mph/Analog - 10-24-2019 11-00-01.410 AM.csv')

# Tares after experiments
wt_load_cell_tare_no_weight_after_experiments = parse_wind_tunnel_csv(
    'data/load_cell_tare_no_weight_after_transition_measurement/Analog - 10-24-2019 11-04-20.174 AM.csv')
wt_load_cell_tare_2oz_weight_after_experiments = parse_wind_tunnel_csv(
    'data/load_cell_tare_2oz_weight_after_transition_measurement/Analog - 10-24-2019 11-06-06.230 AM.csv')
wt_load_cell_tare_8oz_weight_after_experiments = parse_wind_tunnel_csv(
    'data/load_cell_tare_8oz_weight_after_transition_measurement/Analog - 10-24-2019 11-05-31.538 AM.csv')


def parse_manometer_csv(filename):
    """
    Reads the manometer data from the CSV files taken during the experiment.
    :param filename: Name of the CSV file to be read
    :return: a tuple of data, as follows:
    ### Manometer Data Format
    ### All of the manometer data files are vectors of length 25 corresponding to the manometer measurement at that pressure probe.=
    """
    raw_data = np.genfromtxt(
        filename,
        delimiter=',',
        skip_header=1
    )

    manometer_40_mph_angle_1 = raw_data[:, 2]
    manometer_40_mph_angle_2 = raw_data[:, 3]
    manometer_60_mph_angle_1 = raw_data[:, 7]
    manometer_60_mph_angle_2 = raw_data[:, 8]

    return manometer_40_mph_angle_1, manometer_40_mph_angle_2, manometer_60_mph_angle_1, manometer_60_mph_angle_2

### Manometer Data Format
### All of the manometer data files are vectors of length 25 corresponding to the manometer measurement at that pressure probe.

# Get the manometer data
manometer_40_mph_angle_1, manometer_40_mph_angle_2, manometer_60_mph_angle_1, manometer_60_mph_angle_2 = parse_manometer_csv(
    'data/manometer_readings.csv'
)
