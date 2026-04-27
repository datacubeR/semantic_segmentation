import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class BaseVisionDataset(Dataset):
    def shape(self, idx):
        img, mask = self[idx]["image"], self[idx]["mask"]
        return img.shape, mask.shape

    def plot(self, idx, cmap="viridis", figsize=(10, 5), normalized=False):

        img, mask = (
            self[idx]["image"].squeeze(-1),
            self[idx]["mask"].squeeze(-1),
        )
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.title("Image")
        if normalized:
            plt.imshow(img.permute(1, 2, 0).numpy())
        else:
            plt.imshow(img.permute(1, 2, 0).numpy().astype("uint8"))

        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask.permute(1, 2, 0).numpy(), cmap=cmap)
        plt.axis("off")
        plt.show()
