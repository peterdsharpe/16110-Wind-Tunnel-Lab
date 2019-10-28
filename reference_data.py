import numpy as np
import torch
import get_raw_data as raw_data
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint
u = pint.UnitRegistry()

meters_per_inch = (1 * u.inch).to("meter").magnitude

chord_meters = 10 * meters_per_inch
area_meters_squared = (np.pi * 2.5**2) * (meters_per_inch)**2