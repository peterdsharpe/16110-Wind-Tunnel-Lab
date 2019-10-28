import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

import direct_force_measurement as dfm
import wake_measurement as wm
import computational as c

style.use("seaborn")
plt.figure()

# Direct force measurement
plt.errorbar(dfm.Re_body.detach().numpy(), dfm.Cd.detach().numpy(), xerr=dfm.Re_error, yerr=dfm.Cd_error, uplims=True, lolims=True,
             xlolims=True, xuplims=True, fmt=':',
             label="Direct Force Measurement")

# Wake measurement
plt.errorbar(wm.reynolds_number.detach().numpy(), wm.Cd.detach().numpy(), xerr=wm.Re_error, yerr=wm.Cd_error, uplims=True,
             lolims=True, xlolims=True, xuplims=True, fmt='.',
             label="Wake Survey Measurement")

# Computational
plt.plot(c.Re, c.Cd[:,0], '-o', label="MTFLOW, Ncrit = 4")
plt.plot(c.Re, c.Cd[:,1], '-s', label="MTFLOW, Ncrit = 9")

plt.xlabel("$Re_c$")
plt.ylabel("$C_D$ ($S_{ref} = S_{frontal}$)")
plt.legend()
plt.title("Drag Coefficient of a NACA 0050 Body of Revolution")

plt.show()