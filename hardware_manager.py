from typing import Dict, Tuple
import pandas as pd


class HardwareManager:
    _hardware_map: Dict[int, str] = {}
    _specs_map: Dict[str, Tuple[int, int]] = {}
    num_hardwares: int = 0

    @classmethod
    def init_manager(cls, data: pd.DataFrame):
        """Initialize hardware mappings and specs from a pandas DataFrame.

        Parameters:
            data (pd.DataFrame): the dataframe containing a "hardware" column
                * the hardware column must have values in the form f"{cpu_count}_{mem_gb}".
                * Ex: "4_16" for a 4 core, 16 GB hardware
        """
        cls._hardware_map, cls._specs_map = cls._load_hardware_map_and_specs(data)
        cls.num_hardwares = len(cls._hardware_map)

    @staticmethod
    def _load_hardware_map_and_specs(df: pd.DataFrame):
        """Load hardware mappings and specs from the data.csv file."""
        hardware_map: Dict[int, str] = {}
        specs_map: Dict[str, Tuple[int, int]] = {}
        try:
            unique_hardware = df.sort_values(by="hardware")[
                "hardware"
            ].unique()  # Get unique hardware values
            hardware_names = {
                value: index for index, value in enumerate(unique_hardware)
            }
            for _, row in df.iterrows():
                hardware_value = row["hardware"]  # e.g., 8_20, 12_20, etc.
                hardware_name = hardware_names[hardware_value]
                # print(f"{hardware_value=}")
                cpu_count = int(hardware_value.split("_")[0])  # Column for CPU count
                memory_gb = int(hardware_value.split("_")[1])  # Column for memory in GB

                hardware_map[hardware_name] = hardware_value
                specs_map[hardware_value] = (cpu_count, memory_gb)
        except Exception as e:
            raise ValueError(f"Failed to load hardware mappings and specs: {e}") from e

        return hardware_map, specs_map

    @classmethod
    def get_hardware(cls, name: int) -> str:
        """Get hardware value string by idx."""
        assert name in cls._hardware_map, f"Invalid hardware: {name}"
        return cls._hardware_map[name]

    @classmethod
    def get_hardware_idx(cls, value: str) -> int:
        """Get hardware idx by value string."""
        for name, hardware_value in cls._hardware_map.items():
            if hardware_value == value:
                return name
        raise AssertionError(f"Invalid hardware value: {value}")

    @classmethod
    def spec_from_hardware(cls, hardware_value: str) -> Tuple[int, int]:
        """Get CPU and memory specs (tuple) from a hardware value string."""
        assert (
            hardware_value in cls._specs_map
        ), f"Invalid hardware value: {hardware_value}"
        return cls._specs_map[hardware_value]

    @classmethod
    def spec_from_hardware_idx(cls, name: int) -> Tuple[int, int]:
        """Get CPU and memory specs (tuple) from a hardware index."""
        return cls.spec_from_hardware(cls.get_hardware(name))
