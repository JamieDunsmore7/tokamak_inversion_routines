import MDSplus as mds
import numpy as np
import matplotlib.pyplot as plt

shot = 1070511010

node = mds.Tree('spectroscopy', shot)
node = node.getNode('\\spectroscopy::top.bolometer.results.diode.'+\
    '{:s}:BRIGHT'.format('LYMID'))




brightness_data = node.data()
R_values = node.dim_of(0).data()
t_values = node.dim_of(1).data()


# plot an example profile
plt.scatter(R_values, brightness_data[300], label='time = ' + str(t_values[300]) + 's')
plt.xlabel('Major radius, R [m]')
plt.ylabel('Brightness [W/m^2]')
plt.title('Ly-alpha brightness profile at t = ' + str(t_values[300]) + 's')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('shot_1070511010_brightness_profile.png')
plt.show()

# plot the time evolution of a single channel
plt.plot(t_values, brightness_data[:, 10], label = 'Radius = ' + str(R_values[10]) + 'm')
plt.xlabel('Time [s]')
plt.ylabel('Brightness [W/m^2]')
plt.title('Ly-alpha brightness time-trace for R = ' + str(R_values[10]) + 'm')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('shot_1070511010_brightness_timetrace.png')
plt.show()



# save the data as text files
np.savetxt('shot_1070511010_brightness_data.txt', brightness_data)
np.savetxt('shot_1070511010_R_values.txt', R_values)
np.savetxt('shot_1070511010_t_values.txt', t_values)


np.save('shot_1070511010_brightness_data.npy', brightness_data)
np.save('shot_1070511010_R_values.npy', R_values)
np.save('shot_1070511010_t_values.npy', t_values)
