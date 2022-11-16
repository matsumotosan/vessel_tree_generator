import os
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import trange

from src.config import VesselConfig
from src.generator import Generator


cs = ConfigStore.instance()
cs.store(name="vessel_config", node=VesselConfig)

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
    
    # Initialize progress bar
    pbar = trange(cfg.n_trees, desc="Generate vessel trees")
    
    # Generate vessels
    for idx in pbar:
        tree_dir = os.path.join(
            generator.paths.save_dir,
            generator.paths.dataset_name
        )
        
        # Generate vessel tree
        pbar.set_description("Generating tree centerline")
        generator.generate_tree()
        generator.save_tree(tree_dir)
        
        # Generate vessel surface
        pbar.set_description("Generating vessel surface")
        generator.generate_surface()
        generator.save_surface(
            os.path.join(tree_dir, f"{idx}_3Dsurface")
        )        
        
        # Generate projection
        pbar.set_description("Generating projection")
        generator.generate_projection()
        generator.save_projection(tree_dir)


if __name__ == "__main__":
    main()
