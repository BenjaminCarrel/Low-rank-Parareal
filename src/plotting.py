# %% IMPORTATIONS
import matplotlib.animation as animation
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt, cm
import scipy.io as sio
import matplotlib as mpl
from low_rank_toolbox.svd import SVD



def semilogy_colored(x, y, linestyles, labels,
                     title=None, xlabel='xs-label', ylabel='y-label',
                     loc='best', fontsize=14, dpi=125, figsize=(7, 5), grid=True):
    """

    Parameters
    ----------
    x: tuple
    y: tuple
    linestyles: tuple
    labels: tuple
    """
    # CHECK TUPLE
    if type(x) is not tuple:
        raise ValueError('xs is not a tuple')
    # PLOT
    fig = plt.figure(dpi=dpi, figsize=figsize)
    n = len(x)
    for i in np.arange(n):
        plt.semilogy(x[i], y[i], linestyles[i], label=labels[i])
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(loc=loc, fontsize=fontsize)
    if grid:
        plt.grid()
    plt.tight_layout()
    plt.show()
    return fig


# %% 3D PLOT
def plot_3D(X, Y, Z, title=None):
    """
    Create a 3D plot in color
    Parameters
    ----------
    X, Y: ndarray
        Use the command np.meshgrid(xs,y) to make it
    Z: ndarray nxn
        Function to plot
    title: str
    do_save: boolean
    """
    fig = plt.figure(clear=True)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.tight_layout()
    # ax.set_zlim((0, 0.01))
    if title is not None:
        plt.title(title, y=1.03, fontsize=16)
    plt.show()
    return fig

#%% COOKIE PLOT
def plot_cookie(time, solution):
    "Plot cookie solution at given time."
    # Import data
    problem_cookie = sio.loadmat('resources/parametric_cookie_2x2_1580.mat')
    # Construct a triangulation. The triangles are indexed by 1,2, ... in MATLAB hence -1
    FreeDofs = problem_cookie['FreeDofs'][0]
    U_bd = problem_cookie['U_bd'].astype(np.float64).reshape(-1)  # we need floating points, not integers
    mesh_elements = problem_cookie['Mesh']['Elements'][0][0]
    mesh_coordinates = problem_cookie['Mesh']['Coordinates'][0][0]
    tri = mpl.tri.Triangulation(x=mesh_coordinates[:, 0], y=mesh_coordinates[:, 1], triangles=mesh_elements - 1)
    # Plot only the triangles
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')  # to have bigger plots
    ax.set_aspect('equal')
    ax.triplot(tri, lw=0.2, color='grey')
    ax.set_title('Triangulation')

    # Combine solution with boundary solution.
    def add_boundary(y):
        u = U_bd.copy()  # just to be sure, copy
        u[FreeDofs - 1] = y  # MATLAB indices start with 1
        return u

    def plot_solution(y, title=None):
        u = add_boundary(y)

        # Now plot. Countourf seems to be the standard choice.
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        ax.set_aspect('equal')
        ax.triplot(tri, lw=0.1, color='white')  # wireframe of mesh
        tcf = ax.tricontourf(tri, u, 30)  # automatic N levels
        tcf = ax.tricontourf(tri, u, levels=np.linspace(0, 1.18, 100))  # hardcoded after finding that max is 1.1779
        fig.colorbar(tcf)
        if title is not None:
            ax.set_title(title)

    plot_solution(solution, title=f'Solution at final time t={time}')
    plt.show()





#%% SINGULAR VALUE
def singular_values(time: list,
                    solution: list,
                    title: str,
                    do_save: bool = False) -> animation.FuncAnimation:
    "Return an animation of the singular values"

    # STORE SINGULAR VALUES
    nb_t_steps = len(time)
    shape = solution[0].shape
    sing_vals = np.zeros([nb_t_steps, min(shape)])
    for j in np.arange(nb_t_steps):
        if isinstance(solution[j], SVD):
            solution[j] = solution[j].todense()
        sing_vals[j] = la.svd(solution[j], compute_uv=False)

    # PLOT SINGULAR VALUES
    fig, ax = plt.subplots(1, dpi=100)
    plot = ax.semilogy(sing_vals[0], 'bo', label='Sing. Values')
    ax.legend()
    plt.grid()
    ax.set(xlabel='Index', ylabel='Value', title=f'Singular values at time t={round(time[0], 3)}')
    plt.close()

    def update_plot(frame_nb, data, plot):
        ax.lines[0].remove()
        plot = ax.semilogy(sing_vals[frame_nb], 'bo')
        ax.set_title(f'Singular values at time t={round(time[frame_nb], 3)}')

    anim_sing_vals = animation.FuncAnimation(fig, update_plot, nb_t_steps, fargs=(sing_vals, plot))
    if do_save:
        anim_sing_vals.save(title, fps=30)

    return anim_sing_vals


#%% 2D ANIMATION PLOT
def animation_2D(time: list,
                 solution: list,
                 title: str,
                 do_save: bool = False) -> animation.FuncAnimation:
    "Return an animation of the solution in 2D"

    # Updating function
    def update_surf(frame_number, sol, plot):
        plot[0].remove()
        plot[0] = plt.imshow(sol[frame_number])
        plot[0].axes.get_xaxis().set_visible(False)
        plot[0].axes.get_yaxis().set_visible(False)
        plt.title(f'Solution at time t = {round(time[frame_number], 3)}')

    # Figure and first image
    fig = plt.figure()
    plot = [plt.imshow(solution[0])]
    plot[0].axes.get_xaxis().set_visible(False)
    plot[0].axes.get_yaxis().set_visible(False)
    plt.title(f'Solution at time t = {time[0]}')
    n_frame = len(solution)
    anim2d = animation.FuncAnimation(fig, update_surf, n_frame, fargs=(solution, plot))
    if do_save:
        anim2d.save(title, fps=30)
    plt.close()
    return anim2d


# %% 3D ANIMATION PLOT
def animation_3D(x, y, sol_to_plot, title, do_save=False):
    """
    Create and save an animation of a time-dependent 2-dimensions solution.

    Parameters
    ----------
    x,y : ndarray
        Vectors of space
    sol_to_plot: ndarray
        Array containg the solution with coordinates (ts,fx,fy)
    title: str
        Title of animation

    Returns
    -------
    anim3D: animation
    """
    nb_t_steps = sol_to_plot.shape[0]

    X, Y = np.meshgrid(x, y)
    Z = sol_to_plot

    # Function of updating
    def update_surf(frame_number, Z, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, Z[frame_number, :, :], cmap=cm.coolwarm)

    # Figure and first image
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot = [ax.plot_surface(X, Y, Z[0, :, :])]
    plt.close()
    anim3d = animation.FuncAnimation(fig, update_surf, nb_t_steps, fargs=(Z, plot))
    if do_save:
        anim3d.save(title, fps=30)
    return anim3d

