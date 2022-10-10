import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fwd_projection_functions import set_axes_equal


def plot_surface(surface_coords, filename=None, show=False) -> None:
    """Plot and optionally save surface coordinates."""
    fig = plt.figure(figsize=(2,2), dpi=200, constrained_layout=True)
    ax = fig.add_subplot(projection=Axes3D.name)
    ax.view_init(elev=20., azim=-70)
    for surf_coords in surface_coords:
        ax.plot_surface(
            surf_coords[:,:,0],
            surf_coords[:,:,1],
            surf_coords[:,:,2],
            alpha=0.5,
            color="blue"
        )
    set_axes_equal(ax)
    plt.axis('off')
    if show:
        plt.show()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()
