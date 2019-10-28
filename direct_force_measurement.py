import numpy as np
import torch
import get_raw_data as raw_data
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

import process_data
import reference_data
import ambient_conditions




speed_in_meters_per_second = torch.sqrt(2 * process_data.q_at_speed_in_Pa / ambient_conditions.rho_kgm3)

Re_body = ambient_conditions.rho_kgm3 * speed_in_meters_per_second * reference_data.chord_meters / ambient_conditions.mu

### Drag of the NACA 0010 from XFoil
Re_xfoil = ambient_conditions.rho_kgm3 * speed_in_meters_per_second * 0.0254*1.25 / ambient_conditions.mu
Cd_strut_strut_chord = torch.tensor([
    0.02217, 0.02069, 0.01958, 0.01776, 0.01645, 0.01545, 0.01518
])
Cd_strut_frontal_area = Cd_strut_strut_chord * (0.0254*1.25 * 0.0254 * 9)/reference_data.area_meters_squared

### Drag calculation
CdA = process_data.forces_at_speed_in_newtons / process_data.q_at_speed_in_Pa
Cd = CdA / reference_data.area_meters_squared - Cd_strut_frontal_area


def root_sum_sqr(vector):
    return torch.sqrt(torch.sum(vector**2))


# Propagate Cd Error
Cd_error = torch.zeros_like(Cd)
for i in range(len(Cd_error)):
    gradient_Cd_voltage = torch.autograd.grad(outputs=Cd[i], inputs=process_data.voltages_at_speed, retain_graph=True)[0]
    gradient_Cd_q = torch.autograd.grad(outputs=Cd[i], inputs = process_data.raw_q_at_speed_in_torr, retain_graph=True)[0]
    Cd_error[i] = root_sum_sqr(torch.cat((
        gradient_Cd_voltage * process_data.stdevs_of_voltages_at_speed,
        gradient_Cd_q * process_data.stdevs_of_raw_q_at_speed_in_torr,
    )))
print("Cd_error:", Cd_error.numpy())

# Propagate Re Error
Re_error = torch.zeros_like(Re_body)
for i in range(len(Re_error)):
    gradient_Re_q = torch.autograd.grad(outputs = Re_body[i], inputs = process_data.raw_q_at_speed_in_torr, retain_graph = True)[0]
    gradient_Re_temperature = torch.autograd.grad(Re_body[i], ambient_conditions.temperature_K, retain_graph=True)[0]
    gradient_Re_pressure = torch.autograd.grad(Re_body[i], ambient_conditions.pressure_Pa, retain_graph=True)[0]
    Re_error[i] = root_sum_sqr(torch.cat((
        gradient_Re_q * process_data.stdevs_of_raw_q_at_speed_in_torr,
        (gradient_Re_temperature * 0.5).reshape(-1),
        (gradient_Re_pressure * 0.01 * ambient_conditions.Pa_per_inHg).reshape(-1)
    )))
print("Re_error:", Re_error.numpy())



# Plot
if __name__ == "__main__":
    style.use('seaborn')
    # plt.plot(Re_body.detach().numpy(), Cd.detach().numpy(), '-o', label = "Direct Force Measurement")
    plt.errorbar(Re_body.detach().numpy(), Cd.detach().numpy(), xerr=Re_error, yerr=Cd_error, uplims=True, lolims=True, xlolims=True, xuplims=True,
                 label="Direct Force Measurement")
    plt.xlabel("$Re_c$")
    plt.ylabel("$C_D$ ($S_{ref} = S_{frontal}$)")
    plt.title("Drag Coefficient: Direct Force Measurement")

    plt.show()
