
'''
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Need to use in order to run on mac
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

#=============================================================================================

t_start = 1917 # start frame
t_end = 2130 # end frame

data = pd.read_csv('Smart-first_phase_NaN-zeros.csv') # only coordinate data
df = data.loc[t_start:t_end,'Shoulder_left_x':'Ankle_right_z']

# Find max and min values for animation ranges
df_minmax = pd.DataFrame(index=list('xyz'),columns=range(2))
for i in list('xyz'):
    c_max = df.filter(regex='_{}'.format(i)).max().max()
    c_min = df.filter(regex='_{}'.format(i)).min().min()
    df_minmax.ix[i] = np.array([c_min,c_max])

df_minmax = 1.3*df_minmax # increase by 30% to make animation look better

df.columns  = np.repeat(range(12),3) # store cols like this for simplicity
N_tag = df.shape[1]/3 # nr of tags used (all)

N_trajectories = N_tag

t = np.linspace(0,data.Time[t_end],df.shape[0]) # pseudo time-vector for first walking activity
x_t = np.zeros(shape=(N_tag,df.shape[0],3)) # empty animation array (3D)

for tag in range(12):
    # store data in numpy 3D array: (tag,time-stamp,xyz-coordinates)
    x_t[tag,:,:] = df[tag]

x_t = x_t[:, :, [0, 2, 1]]

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('on')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))
# set up trajectory lines
lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
# set up points
pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])
# set up lines which create the stick figures

stick_defines = [
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11)
]

stick_lines = [ax.plot([], [], [], 'k-')[0] for _ in stick_defines]

# prepare the axes limits
ax.set_xlim(df_minmax.ix['x'].values)
ax.set_ylim(df_minmax.ix['z'].values) # note usage of z coordinate
ax.set_zlim(df_minmax.ix['y'].values) # note usage of y coordinate

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        # trajectory lines
        line.set_data([], [])
        line.set_3d_properties([])
        # points
        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts + stick_lines

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (5 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T # note ordering of points to line up with true exogenous registration (x,z,y)
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
        stick_line._verts3d = x_t[[sp,ep], i, :].T.tolist()

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts + stick_lines

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=30, blit=True)

plt.show()


'''
import matplotlib.pyplot as plt
import numpy as np
import time

fig, ax = plt.subplots()
line, = ax.plot(np.random.randn(100))
plt.show(block=False)

tstart = time.time()
num_plots = 0

while time.time()-tstart < 5:
    line.set_ydata(np.random.randn(100))
    ax.draw_artist(ax.patch)
    ax.draw_artist(line)
    #fig.canvas.update()
    fig.canvas.flush_events()
    num_plots += 1
    plt.pause(0.001)
print(num_plots/5)