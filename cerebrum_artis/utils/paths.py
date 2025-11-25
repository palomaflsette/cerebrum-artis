"""
Path Configuration Utilities
=============================

Manages all paths for data, checkpoints, and outputs.
Uses environment variables when available, falls back to defaults.
"""

import os
from pathlib import Path
import dotenv
dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_data_path():
    """Get path to ArtEmis dataset directory."""
    env_path = os.getenv('ARTEMIS_DATA_PATH')
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / 'data' / 'artemis' / 'dataset' / 'official_data'


def get_image_dir():
    """Get path to WikiArt images directory."""
    env_path = os.getenv('WIKIART_IMAGE_DIR')
    if env_path:
        return Path(env_path)
    return None


def get_checkpoint_dir():
    """Get path to model checkpoints directory."""
    env_path = os.getenv('CHECKPOINT_DIR')
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / 'checkpoints'


def get_fuzzy_cache_path():
    """Get path to fuzzy features cache file."""
    env_path = os.getenv('FUZZY_CACHE_PATH')
    if env_path:
        return Path(env_path)
    return None


def get_output_dir():
    """Get path to outputs directory (for notebooks)."""
    env_path = os.getenv('OUTPUT_DIR')
    if env_path:
        return Path(env_path)
    return PROJECT_ROOT / 'outputs'


def get_artemis_csv():
    """Get full path to ArtEmis CSV file."""
    return get_data_path() / 'combined_artemis_with_splits.csv'


class PathConfig:
    """Configuration object with all paths."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = get_data_path()
        self.image_dir = get_image_dir()
        self.checkpoint_dir = get_checkpoint_dir()
        self.fuzzy_cache = get_fuzzy_cache_path()
        self.output_dir = get_output_dir()
        self.artemis_csv = get_artemis_csv()
    
    def __repr__(self):
        return (
            f"PathConfig(\n"
            f"  project_root='{self.project_root}',\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  image_dir='{self.image_dir}',\n"
            f"  checkpoint_dir='{self.checkpoint_dir}',\n"
            f"  fuzzy_cache='{self.fuzzy_cache}',\n"
            f"  output_dir='{self.output_dir}',\n"
            f"  artemis_csv='{self.artemis_csv}'\n"
            f")"
        )
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("="*60)
        print("PATH CONFIGURATION")
        print("="*60)
        print(f"Project Root:    {self.project_root}")
        print(f"Data Dir:        {self.data_dir}")
        print(f"Image Dir:       {self.image_dir}")
        print(f"Checkpoint Dir:  {self.checkpoint_dir}")
        print(f"Fuzzy Cache:     {self.fuzzy_cache}")
        print(f"Output Dir:      {self.output_dir}")
        print(f"ArtEmis CSV:     {self.artemis_csv}")
        print("="*60)


paths = PathConfig()


if __name__ == '__main__':
    paths.print_config()
