from pathlib import Path

import yaml


class ConfigLoader:

    # Input: Path to a YAML configuration file.
    # Output: Dictionary containing the configuration settings.
    # Purpose: Load and validate project settings from a YAML file.
    def load(self, config_path):

        config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration must contain a YAML dictionary."
            )

        required_sections = [
            "video",
            "model",
            "counting",
            "analytics"
        ]

        for section in required_sections:
            if section not in config or not isinstance(config[section], dict):
                raise ValueError(
                    f"Missing or invalid configuration section: {section}"
                )

        return config