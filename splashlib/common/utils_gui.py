import ipyvolume as ipv
import matplotlib.pyplot as plt
import numpy as np
from splashlib.common.utils_data import constrict_ranges


# install ipyvolume with conda install -c conda-forge ipyvolume

def plot_3d_scatter_animation(x, y, z, 
                            color=None, 
                            size=1, 
                            interval=500, 
                            interpolate=False, 
                            xlim=None, 
                            ylim=None, 
                            zlim=None):
    """
    Plot an animation of scatter points. Refer to ipyvolume documentation for the values.
    I suggest not to use interpolate as it gives weird results.
    Also I suggest setting the limits otherwise the point clouds will not appear realistic 
    (i.e the bounding box will match the point clouds and thus appear stretched).
    """
    fig = ipv.figure()
    if color is not None:
        scatter = ipv.scatter(x, y, z, color=color, size=size, marker="sphere")
    else:
        scatter = ipv.scatter(x, y, z, size=size, marker="sphere")
    ipv.animation_control(scatter, interval=interval)
    if xlim is not None:
        ipv.xlim(xlim[0], xlim[1])
    if ylim is not None:
        ipv.ylim(ylim[0], ylim[1])
    if zlim is not None:
        ipv.zlim(zlim[0], zlim[1])
    ipv.show()
    if not interpolate:
        fig.animation = 0
    return fig, scatter


def plot_3d_scatter_frame(x, y, z, color=None, size=1, xlim=None, ylim=None, zlim=None):
    """
    Plot a single frame of scatter points. Refer to ipyvolume documentation for the values.
    I suggest setting the limits otherwise the point clouds will not appear realistic 
    (i.e the bounding box will match the point clouds and thus appear stretched).
    """
    fig = ipv.figure()
    if color is not None:
        scatter = ipv.scatter(x, y, z, size=size, marker="sphere", color=color)
    else:
        scatter = ipv.scatter(x, y, z, size=size, marker="sphere")
    if xlim is not None:
        ipv.xlim(xlim[0], xlim[1])
    if ylim is not None:
        ipv.ylim(ylim[0], ylim[1])
    if zlim is not None:
        ipv.zlim(zlim[0], zlim[1])
    ipv.show()
    return fig, scatter

def normalize_01(ar):
    """
    Normalize the input array to a [0,1] range.
    """
    v = ar.copy()
    v_min = np.min(v)
    v = v - v_min
    v_max = np.max(v)
    return v/v_max

def ext_points(xyz_all, color_map = 'coolwarm', index_to_repeat=-1, xlim=None, ylim=None, zlim=None, max_num=None):
    """
    Given an array of x,y,z coordinates, split it into the correpsonding sub arrays + velocity and use velocity as color.

    xyz_all: input array
    color_map: color map to use
    index_to_repeat: as we just need to plot, we replicate one point multiple times so that all frames have 
        the same number of points (similar to replication_strategy in utils_data.refill_dataset). 
        index_to_repeat is the index of the point to repeat.
    xlim: tuple (from, to) to set the x limit
    ylim: tuple (from, to) to set the y limit
    zlim: tuple (from, to) to set the z limit
    max_num: hardcoded maximum number of points to show in each frame. Points are capped, rather than sampled at random.
    """

    x_s = []
    y_s = []
    z_s = []
    colors = []
    cmap = plt.get_cmap(color_map)
    local_max_num = 0 # max number of points in a frame: replicate last point so that all frames in a sequence have the same number of points

    for xyz in xyz_all:

        if len(xyz.shape) > 1:
            move_id = np.where(abs(xyz[:,3])>=0)

            x = xyz[move_id, 0].flatten()
            y = xyz[move_id, 1].flatten()
            z = xyz[move_id, 2].flatten()
            v = xyz[move_id, 3].flatten() # doppler velocity
        else:
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            v = xyz[3]
        
        # remove points outside boundaries
        x, y, z, mask = constrict_ranges(x, y, z, xlim, ylim, zlim)
        if not isinstance(v, np.ndarray):
            v = np.array([v])
        v = v[mask]
        
        c = normalize_01(v)
        color = np.array(cmap(c))
        
        if len(x) > local_max_num:
            local_max_num = len(x)
        
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        colors.append(color)
    
    if max_num is not None and local_max_num > max_num:
        print(f"WARNING: There are more points than the predefined {max_num}")
    elif max_num is None:
        max_num = local_max_num
        
    for i in range(len(x_s)):
        current_len = len(x_s[i])
        if current_len < max_num:
            x_s[i] = np.append(x_s[i], [x_s[i][index_to_repeat]] * (max_num - current_len))
            y_s[i] = np.append(y_s[i], [y_s[i][index_to_repeat]] * (max_num - current_len))
            z_s[i] = np.append(z_s[i], [z_s[i][index_to_repeat]] * (max_num - current_len))
            colors[i] = np.append(colors[i], np.tile(colors[i][index_to_repeat,:] ,((max_num - current_len), 1)), axis=0)
        elif current_len > max_num:
            # remove last points
            x_s[i] = x_s[i][:max_num]
            y_s[i] = y_s[i][:max_num]
            z_s[i] = z_s[i][:max_num]
            colors[i] = colors[i][:max_num]

    x_s = np.array(x_s, dtype=float)
    y_s = np.array(y_s, dtype=float)
    z_s = np.array(z_s, dtype=float)
    colors = np.array(colors, dtype=float)

    return x_s, y_s, z_s, colors

def plot_loss_history(history):
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
