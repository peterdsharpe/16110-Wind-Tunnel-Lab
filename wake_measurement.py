import numpy as np
import torch
import get_raw_data as raw_data
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

import process_data
import reference_data
import ambient_conditions


def root_sum_sqr(vector):
    return torch.sqrt(torch.sum(vector ** 2))


dynamic_pressure_torr = torch.tensor([
    1.473, 3.278  # Check data
])
dynamic_pressure_Pa = dynamic_pressure_torr * process_data.pascals_per_torr

speed_in_meters_per_second = torch.sqrt(2 * dynamic_pressure_Pa / ambient_conditions.rho_kgm3)
reynolds_number = ambient_conditions.rho_kgm3 * speed_in_meters_per_second * reference_data.chord_meters / ambient_conditions.mu

### Process the data
manometer_40_mph_angle_1 = torch.tensor(raw_data.manometer_40_mph_angle_1, requires_grad=True)
manometer_40_mph_angle_2 = torch.tensor(raw_data.manometer_40_mph_angle_2, requires_grad=True)
manometer_60_mph_angle_1 = torch.tensor(raw_data.manometer_60_mph_angle_1, requires_grad=True)
manometer_60_mph_angle_2 = torch.tensor(raw_data.manometer_60_mph_angle_2, requires_grad=True)

# Do the 40 mph case
manometer_40_mph = torch.sum(torch.stack((
    manometer_40_mph_angle_1,
    manometer_40_mph_angle_2,
)), dim=0) / 2
manometer_40_mph_radial = torch.sum(torch.stack((
    torch.flip(manometer_40_mph[1:13], dims=[0]),
    manometer_40_mph[12:-1]
)), dim=0) / 2
uz_over_ue_40 = torch.sqrt(
    (manometer_40_mph_radial - manometer_40_mph[0]) /
    (manometer_40_mph_radial[-1] - manometer_40_mph[0])
)
radial_centers = torch.arange(0, 0.0254 * 0.150 * 13, 0.0254 * 0.150)
radial_break_points = torch.cat((
    torch.tensor([0.]),
    (radial_centers[1:] + radial_centers[:-1]) / 2
))
theta_x_40 = torch.sum(
    (1 - uz_over_ue_40) * uz_over_ue_40 *
    (np.pi * radial_break_points[1:] ** 2 - np.pi * radial_break_points[:-1] ** 2)
)
dstar_x_40 = torch.sum(
    (1 - uz_over_ue_40) *
    (np.pi * radial_break_points[1:] ** 2 - np.pi * radial_break_points[:-1] ** 2)
)
H_x_40 = dstar_x_40 / theta_x_40
ue_over_V_inf_40 = torch.sqrt(
    (manometer_40_mph_radial[-1] - manometer_40_mph[0]) /
    (manometer_40_mph_radial[-1] - manometer_40_mph[-1])
)
theta_inf_40 = theta_x_40 * (ue_over_V_inf_40) ** ((H_x_40 + 5) / 2)

Cd_40 = 2 * theta_inf_40 / reference_data.area_meters_squared

# Do the 60 mph case
manometer_60_mph = torch.sum(torch.stack((
    manometer_60_mph_angle_1,
    manometer_60_mph_angle_2,
)), dim=0) / 2
manometer_60_mph_radial = torch.sum(torch.stack((
    torch.flip(manometer_60_mph[1:13], dims=[0]),
    manometer_60_mph[12:-1]
)), dim=0) / 2
uz_over_ue_60 = torch.sqrt(
    (manometer_60_mph_radial - manometer_60_mph[0]) /
    (manometer_60_mph_radial[-1] - manometer_60_mph[0])
)
radial_centers = torch.arange(0, 0.0254 * 0.150 * 13, 0.0254 * 0.150)
radial_break_points = torch.cat((
    torch.tensor([0.]),
    (radial_centers[1:] + radial_centers[:-1]) / 2
))
theta_x_60 = torch.sum(
    (1 - uz_over_ue_60) * uz_over_ue_60 *
    (np.pi * radial_break_points[1:] ** 2 - np.pi * radial_break_points[:-1] ** 2)
)
dstar_x_60 = torch.sum(
    (1 - uz_over_ue_60) *
    (np.pi * radial_break_points[1:] ** 2 - np.pi * radial_break_points[:-1] ** 2)
)
H_x_60 = dstar_x_60 / theta_x_60
ue_over_V_inf_60 = torch.sqrt(
    (manometer_60_mph_radial[-1] - manometer_60_mph[0]) /
    (manometer_60_mph_radial[-1] - manometer_60_mph[-1])
)
theta_inf_60 = theta_x_60 * (ue_over_V_inf_60) ** ((H_x_60 + 5) / 2)

Cd_60 = 2 * theta_inf_60 / reference_data.area_meters_squared

# Make the requested plots
if __name__ == "__main__":
    r_over_R_max = np.arange(12)/11

    manometer_40_mph_angle_1_radial = torch.sum(torch.stack((
        torch.flip(manometer_40_mph_angle_1[1:13], dims=[0]),
        manometer_40_mph_angle_1[12:-1]
    )), dim=0) / 2
    uz_over_ue_40_angle_1 = torch.sqrt(
        (manometer_40_mph_angle_1_radial - manometer_40_mph_angle_1[0]) /
        (manometer_40_mph_angle_1_radial[-1] - manometer_40_mph_angle_1[0])
    )
    manometer_40_mph_angle_2_radial = torch.sum(torch.stack((
        torch.flip(manometer_40_mph_angle_2[1:13], dims=[0]),
        manometer_40_mph_angle_2[12:-1]
    )), dim=0) / 2
    uz_over_ue_40_angle_2 = torch.sqrt(
        (manometer_40_mph_angle_2_radial - manometer_40_mph_angle_2[0]) /
        (manometer_40_mph_angle_2_radial[-1] - manometer_40_mph_angle_2[0])
    )

    manometer_60_mph_angle_1_radial = torch.sum(torch.stack((
        torch.flip(manometer_60_mph_angle_1[1:13], dims=[0]),
        manometer_60_mph_angle_1[12:-1]
    )), dim=0) / 2
    uz_over_ue_60_angle_1 = torch.sqrt(
        (manometer_60_mph_angle_1_radial - manometer_60_mph_angle_1[0]) /
        (manometer_60_mph_angle_1_radial[-1] - manometer_60_mph_angle_1[0])
    )
    manometer_60_mph_angle_2_radial = torch.sum(torch.stack((
        torch.flip(manometer_60_mph_angle_2[1:13], dims=[0]),
        manometer_60_mph_angle_2[12:-1]
    )), dim=0) / 2
    uz_over_ue_60_angle_2 = torch.sqrt(
        (manometer_60_mph_angle_2_radial - manometer_60_mph_angle_2[0]) /
        (manometer_60_mph_angle_2_radial[-1] - manometer_60_mph_angle_2[0])
    )
    

    plt.figure()
    plt.plot(r_over_R_max, uz_over_ue_40_angle_1.detach().numpy(), '-o', label="Angle 1")
    plt.plot(r_over_R_max, uz_over_ue_40_angle_2.detach().numpy(), '-s', label="Angle 2")
    plt.xlabel("$R/R_{max}$")
    plt.ylabel("$u/u_e$")
    plt.legend()
    plt.title("Wake Velocity Profile, 40 mph")
    plt.show()

    plt.figure()
    plt.plot(r_over_R_max, uz_over_ue_60_angle_1.detach().numpy(), '-o', label="Angle 1")
    plt.plot(r_over_R_max, uz_over_ue_60_angle_2.detach().numpy(), '-s', label="Angle 2")
    plt.xlabel("$R/R_{max}$")
    plt.ylabel("$u/u_e$")
    plt.legend()
    plt.title("Wake Velocity Profile, 60 mph")
    plt.show()

    

# Propagate Cd_40 Error
gradient_Cd_40_manometer_1 = torch.autograd.grad(outputs=Cd_40, inputs=manometer_40_mph_angle_1, retain_graph=True)[0]
gradient_Cd_40_manometer_2 = torch.autograd.grad(outputs=Cd_40, inputs=manometer_40_mph_angle_2, retain_graph=True)[0]
Cd_40_error = root_sum_sqr(torch.cat((
    gradient_Cd_40_manometer_1 * 0.05,
    gradient_Cd_40_manometer_2 * 0.05,
)))
print("Cd_error:", Cd_40_error.numpy())

# Propagate Cd_60 Error
gradient_Cd_60_manometer_1 = torch.autograd.grad(outputs=Cd_60, inputs=manometer_60_mph_angle_1, retain_graph=True)[0]
gradient_Cd_60_manometer_2 = torch.autograd.grad(outputs=Cd_60, inputs=manometer_60_mph_angle_2, retain_graph=True)[0]
Cd_60_error = root_sum_sqr(torch.cat((
    gradient_Cd_60_manometer_1 * 0.05,
    gradient_Cd_60_manometer_2 * 0.05,
)))
print("Cd_error:", Cd_60_error.numpy())

# Propagate Re Error
Re_error = torch.zeros_like(reynolds_number)
for i in range(len(Re_error)):
    gradient_Re_temperature = torch.autograd.grad(reynolds_number[i], ambient_conditions.temperature_K, retain_graph=True)[0]
    gradient_Re_pressure = torch.autograd.grad(reynolds_number[i], ambient_conditions.pressure_Pa, retain_graph=True)[0]
    Re_error[i] = root_sum_sqr(torch.cat((
        (gradient_Re_temperature * 0.5).reshape(-1),
        (gradient_Re_pressure * 0.01 * ambient_conditions.Pa_per_inHg).reshape(-1)
    )))
print("Re_error:", Re_error.numpy())

# Merge things
Cd = torch.stack((Cd_40, Cd_60))
Cd_error = torch.stack((Cd_40_error, Cd_60_error))

# Plot
if __name__ == "__main__":
    style.use('seaborn')
    plt.errorbar(reynolds_number.detach().numpy(), Cd.detach().numpy(), xerr=Re_error, yerr=Cd_error, uplims=True, lolims=True, xlolims=True, xuplims=True,
                 label="Wake Survey Measurement")
    plt.xlabel("$Re_c$")
    plt.ylabel("$C_D$ ($S_{ref} = S_{frontal}$)")
    plt.title("Drag Coefficient: Wake Survey Measurement")

    plt.show()
