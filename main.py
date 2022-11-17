import os
import hydra

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import trange

from src.config import VesselConfig
from src.generator import Generator

cs = ConfigStore.instance()
cs.store(name="vessel_config", node=VesselConfig)

MSG_WIDTH = 35


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: VesselConfig) -> None:
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize Generator
    generator = Generator(
        cfg.paths,
        cfg.flags,
        cfg.geometry,
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
    pbar = trange(cfg.n_trees, desc="Generating vessel trees")

    # Generate vessels
    for idx in pbar:
        # Generate vessel centerlines
        pbar.set_description(f"Generating vessel {idx:04d} centerline".ljust(MSG_WIDTH))
        generator.generate_tree()
        if cfg.flags.save_specs:
            generator.save_tree(
                os.path.join(tree_dir, "specs", f"vessel_{idx:04d}_specs")
            )

        # Generate vessel surface
        pbar.set_description(f"Generating vessel {idx:04d} surface".ljust(MSG_WIDTH))
        generator.generate_surface()
        generator.save_surface(
            os.path.join(tree_dir, "surface", f"vessel_{idx:04d}_surface"),
            cfg.flags.plot_surface
        )

        # Generate projections
        if cfg.flags.generate_projections:
            pbar.set_description(f"Generating vessel {idx:04d} projections".ljust(MSG_WIDTH))
            generator.generate_projections(idx)


if __name__ == "__main__":
    main()
