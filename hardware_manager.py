import pandas as pd
import pathlib
from enum import Enum

class HardwareManager:
    _hardware_map = {}
    _specs_map = {}

    @classmethod
    def init_manager(cls, csv_path):
        """Initialize hardware mappings and specs from a CSV file."""
        cls._hardware_map, cls._specs_map = cls._load_hardware_map_and_specs(csv_path)

    @staticmethod
    def _load_hardware_map_and_specs(csv_path):
        """Load hardware mappings and specs from the data.csv file."""
        hardware_map = {}
        specs_map = {}
        try:
            df = pd.read_csv(csv_path)
            unique_hardware = df["hardware"].unique()  # Get unique hardware values
            hardware_names = {value: index for index, value in enumerate(unique_hardware)}
            for _, row in df.iterrows():
                hardware_value = row["hardware"]  # e.g., 8_20, 12_20, etc.
                hardware_name = hardware_names[hardware_value]
                cpu_count = int(hardware_value.split("_")[0])  # Column for CPU count
                memory_gb = float(hardware_value.split("_")[1])  # Column for memory in GB

                hardware_map[hardware_name] = hardware_value
                specs_map[hardware_value] = (cpu_count, memory_gb)
        except Exception as e:
            raise ValueError(f"Failed to load hardware mappings and specs: {e}")
        
        return hardware_map, specs_map

    @classmethod
    def get_hardware(cls, name):
        """Get hardware value by idx."""
        return cls._hardware_map.get(name)
    
    @classmethod
    def get_hardware_idx(cls, value):
        """Get hardware idx by value."""
        for name, hardware_value in cls._hardware_map.items():
            if hardware_value == value:
                return name
        return None

    @classmethod
    def spec_from_hardware(cls, hardware_value):
        """Get CPU and memory specs from a hardware value."""
        return cls._specs_map.get(hardware_value, (None, None))
