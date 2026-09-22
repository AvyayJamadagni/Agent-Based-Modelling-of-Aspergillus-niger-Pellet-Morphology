import matplotlib.pyplot as plt
import numpy as np

myc_density = 0.15 #g DM / cm3
#growth_rate = 0.18 #g DM / g sucrose
growth_factor = 0.09 
sim_steps = 200
ca_prod_rate = 14 #mg / g DM / h

#growth_volume = growth_rate/myc_density 
v_0 = 0.4
vol_vals = []
ca_vals = []

for step in range(sim_steps):
    vol = v_0 * np.exp(growth_factor*step) #um3
    vol = vol * 1e-12 #cm3
    vol_vals.append(vol)
    dm = vol * myc_density
    ca_prod = dm * ca_prod_rate/1000
    if step == 0:
        ca_vals.append(ca_prod)
    else:
        ca_vals.append(ca_vals[-1] + ca_prod)


plt.figure(figsize=(8, 5))
plt.plot(range(sim_steps), vol_vals, marker='o', label='Mycelium biomass')
plt.plot(range(sim_steps), ca_vals, marker='s', label='Cumulative CA Production')
plt.xlabel('Time Steps (h)')
plt.ylabel('Concentration (g/L)')
plt.title('Mycelium Growth and CA Production Over Time')
plt.legend()
plt.grid()
plt.show()
    
    










