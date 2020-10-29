import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81


def derive(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta_1, z1, theta_2, z2 = y

    c, s = np.cos(theta_1 - theta_2), np.sin(theta_1 - theta_2)

    theta1dot = z1

    z1dot = (m2 * g * np.sin(theta_2) * c
             - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2)
             - (m1 + m2) * g * np.sin(theta_1)) / L1 / (m1 + m2 * s ** 2)

    theta2dot = z2

    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta_2) + g * np.sin(theta_1) * c) +
             m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)

    return theta1dot, z1dot, theta2dot, z2dot


def total_energy(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T

    potential = -(m1 + m2) * L1 * g * np.cos(th1) - m2 * L2 * g * np.cos(th2)

    kinetic = 0.5 * m1 * (L1 * th1d) ** 2 + 0.5 * m2 * ((L1 * th1d) ** 2
        + (L2 * th2d) ** 2 + 2 * L1 * L2 * th1d * th2d * np.cos(th1 - th2))

    return potential + kinetic


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 30, 0.01

t = np.arange(0, tmax + dt, dt)

# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

# Do the numerical integration of the equations of motion
y = odeint(derive, y0, t, args=(L1, L2, m1, m2))

# Check that the calculation conserves total energy to within some tolerance.
EDRIFT = 0.05

# Total energy from the initial conditions
E = total_energy(y0)
if np.max(np.sum(np.abs(total_energy(y) - E))) > EDRIFT:
    sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

# Unpack z and theta as a function of time
theta_1, theta_2 = y[:, 0], y[:, 2]

# Convert to Cartesian coordinates of the two bob positions.
x1 = L1 * np.sin(theta_1)
y1 = -L1 * np.cos(theta_1)
x2 = x1 + L2 * np.sin(theta_2)
y2 = y1 - L2 * np.cos(theta_2)

# Plotted bob circle radius
r = 0.05
# Plot a trail of the m2 bob's position for the last trail_secs seconds.
trail_secs = 0.5
# This corresponds to max_trail time points.
max_trail = int(trail_secs / dt)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
x_data = []
y_data = []
line, = ax.plot(x_data, y_data, lw=2)
ax.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], lw=2, c='k')


def make_plot(i):
    plt.cla()
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')

    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt', lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    plt.axis('off')

    return ax.plot()


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, make_plot, init_func=init,
                               frames=500, interval=1, blit=True)

anim.save('double_pendulum.gif', fps=30)
plt.show()
