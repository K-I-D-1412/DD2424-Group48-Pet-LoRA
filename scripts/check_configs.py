"""
Check that all YAML configuration files can be parsed.
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def main():
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))

    if not config_files:
        raise FileNotFoundError("No YAML config files found in configs/.")

    print(f"Found {len(config_files)} config files.")

    for config_path in config_files:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Config is not a dictionary: {config_path}")

        print(f"OK: {config_path.name}")

    print("All config files parsed successfully.")


if __name__ == "__main__":
    main()