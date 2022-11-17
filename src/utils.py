import json
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join, exists
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

from src.fwd_projection_functions import set_axes_equal


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


def create_nested_dir(folder, sub_folder):
    if not exists(folder):
        makedirs(folder)
    if not exists(join(folder, sub_folder)):
        makedirs(join(folder, sub_folder))


def save_specs(json_path, json_obj) -> None:
    with open(json_path, 'w+') as f:
        json.dump(json_obj, f, indent=2)


def add_general_parser_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--save_path', default=None, type=str, required=True)
    parser.add_argument('--dataset_name', default="test", type=str)
    parser.add_argument('--num_trees', default=10, type=int)
    parser.add_argument('--save_visualization', action='store_true', help="this flag will plot the generated 3D surfaces and save it as a PNG")
    return parser

def add_centerline_parser_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--num_branches', default=0, type=int,
                        help="Number of side branches. Set to 0 for no side branches")
    parser.add_argument('--vessel_type', default="RCA", type=str, help="Options are: 'cylinder, 'spline', and 'RCA'")
    parser.add_argument('--control_point_path', default="./RCA_branch_control_points/moderate", type=str)
    parser.add_argument('--num_centerline_points', default=200, type=int)
    parser.add_argument('--centerline_supersampling', default=1, type=int, help="factor by which to super-sample centerline points when generating vessel surface")
    parser.add_argument('--shear', action='store_true', help="add random shear augmentation")
    parser.add_argument('--warp', action='store_true', help="add random warping augmentation")
    return parser

def add_stenosis_parser_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--constant_radius', action='store_true')
    parser.add_argument('--num_stenoses', default=None, type=int)
    parser.add_argument('--stenosis_position', nargs="*", default=None, type=int)
    parser.add_argument('--stenosis_severity', nargs="*", default=None, type=float)
    parser.add_argument('--stenosis_length', nargs="*", default=None, type=int, help="number of points in radius vector where stenosis will be introduced")
    return parser

def add_projection_parser_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--generate_projections', action="store_true")
    parser.add_argument('--num_projections', default=3, type=int,
                        help="number of random projection images to generate")
    return parser