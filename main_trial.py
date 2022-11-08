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
    
    pbar = trange(cfg.n_trees, desc="Generate vessel trees")
    for idx in pbar:
        # Generate vessel tree
        pbar.set_description("Generating tree")
        generator.generate_tree()

        pbar.set_description("Saving tree")
        generator.save_tree()
        
        # Generate projection
        pbar.set_description("Generating projection")
        generator.generate_projection()
        
        pbar.set_description("Saving projection")
        generator.save_projection()
    

if __name__ == "__main__":
    main()
