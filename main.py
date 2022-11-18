import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import trange

from src.config import VesselConfig
from src.generator import Generator

cs = ConfigStore.instance()
cs.store(name="vessel_config", node=VesselConfig)

MSG_WIDTH = 34


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: VesselConfig) -> None:
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize Generator
    generator = Generator(
        cfg.paths,
        cfg.flags,
        cfg.geometry,
        cfg.tree,
        cfg.projections
    )

    # Specify directory to save outputs
    tree_dir = os.path.join(
        generator.paths.save_dir,
        generator.paths.dataset_name
    )

    # Prepare directories
    generator.prepare_dirs(tree_dir)

    # Initialize progress bar
    pbar = trange(cfg.flags.n_vessels)

    # Generate vessels
    for idx in pbar:
        # Generate vessel centerlines
        pbar.set_description(f"Generating vessel {idx:04d} centerline".ljust(MSG_WIDTH))
        generator.generate_tree()
        
        # Save vessel generation specs
        # generator.save_specs(
        #     os.path.join(tree_dir, "specs", f"vessel_{idx:04d}_specs")
        # )

        # Generate vessel surface
        pbar.set_description(f"Generating vessel {idx:04d} surface".ljust(MSG_WIDTH))
        generator.generate_surface()
        
        # Save vessel surface coordinates
        filename = os.path.join(tree_dir, "surface", f"vessel_{idx:04d}_surface")
        if cfg.flags.split_by_branch:
            filename += "_bybranch"
        generator.save_surface(
            filename,
            split_by_branch=cfg.flags.split_by_branch
        )
        
        # Save vessel surface plot
        if cfg.flags.save_surface_plot:
            generator.save_surface_plot(
                os.path.join(tree_dir, "surface", f"vessel_{idx:04d}_surface")
            )

        # Generate projections
        if cfg.flags.generate_projections:
            pbar.set_description(f"Generating vessel {idx:04d} projections".ljust(MSG_WIDTH))
            generator.generate_projections(idx)

    print(f"Finished generating {cfg.n_trees} vessels.")


if __name__ == "__main__":
    main()