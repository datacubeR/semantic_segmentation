from .basers import BaseRSDataset


class FullImageDataset(BaseRSDataset):
    @property
    def get_deadtrees_class_names(self):
        class_names = ["Non-Dead Tree", "Dead Tree"]
        return class_names

    @property
    def get_loveda_class_names(self):
        class_names = [
            "Ignore",
            "Background",
            "Building",
            "Road",
            "Water",
            "Barren",
            "Forest",
            "Agricultural land",
        ]
        return class_names
