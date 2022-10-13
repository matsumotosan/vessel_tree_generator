import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm

from src.config import VesselConfig
from src.generator import Generator


cs = ConfigStore.instance()
cs.store(name="vessel_config", node=VesselConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: VesselConfig) -> None:
    print(f"Generating vessels with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize vessel Generator object
    generator = Generator(
        cfg.paths,
        cfg.geometry,
        cfg.projections,
        cfg.flags
    )
    
    for idx in tqdm(range(cfg.n_trees)):
        # Generate vessel tree
        generator.generate_tree()
        generator.save_tree()
        
        # Generate projection
        generator.generate_projection()
        generator.save_projection()
    
    