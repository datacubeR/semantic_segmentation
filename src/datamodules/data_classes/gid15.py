from .basers import BaseRSDataset


class GID15(BaseRSDataset):
    @property
    def get_class_names(self):
        gid15_class_names = [
            "background",  # 0
            "industrial_land",  # 1
            "urban_residential",  # 2
            "rural_residential",  # 3
            "traffic_land",  # 4
            "paddy_field",  # 5
            "irrigated_land",  # 6
            "dry_cropland",  # 7
            "garden_plot",  # 8
            "arbor_woodland",  # 9
            "shrub_land",  # 10
            "natural_grassland",  # 11
            "artificial_grassland",  # 12
            "river",  # 13
            "lake",  # 14
            "pond",  # 15
        ]
        return gid15_class_names
