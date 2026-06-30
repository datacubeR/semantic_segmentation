import gc

import torchio as tio
from datasets import Array2D, Array3D, Dataset, Features


class HFDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def build_patches(
        self,
        output_path: str,
        patch_kwargs: dict = None,
        shard_size: int = 5000,
    ):
        all_images = []
        all_masks = []

        patch_kwargs["patch_size"][0]
        windows_params = dict(
            ph=patch_kwargs["patch_size"][0],
            pw=patch_kwargs["patch_size"][1],
            oh=patch_kwargs["overlap"][0],
            ow=patch_kwargs["overlap"][1],
        )

        shard_idx = 0
        for element in self.dataset:
            image = element["image"].unsqueeze(-1).squeeze(0)
            mask = element["mask"].unsqueeze(-1)
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image), mask=tio.LabelMap(tensor=mask)
            )
            sampler = tio.GridSampler(
                subject,
                patch_size=(windows_params["ph"], windows_params["pw"], 1),
                patch_overlap=(windows_params["oh"], windows_params["ow"], 0),
            )
            for patch in sampler:
                img_np = (
                    patch["image"][tio.DATA]
                    .squeeze(-1)
                    .contiguous()
                    .numpy()
                    .astype("float32")
                )
                msk_np = (
                    patch["mask"][tio.DATA]
                    .squeeze(-1)
                    .squeeze(0)
                    .contiguous()
                    .numpy()
                    .astype("uint8")
                )
                del patch

                all_images.append(img_np)
                all_masks.append(msk_np)

                if len(all_images) >= shard_size:
                    self._save_shard(
                        output_path, shard_idx, all_images, all_masks, windows_params
                    )
                    shard_idx += 1

                    all_images.clear()
                    all_masks.clear()

                    gc.collect()

            del subject, sampler
            gc.collect()

        if len(all_images) > 0:
            self._save_shard(
                output_path, shard_idx, all_images, all_masks, windows_params
            )
            all_images.clear()
            all_masks.clear()
            gc.collect()

    def build_full(self, output_path: str, shard_size: int = 2):
        all_images = []
        all_masks = []

        shard_idx = 0

        for element in self.dataset:
            image = element["image"].squeeze(0)
            mask = element["mask"]

            all_images.append(image.numpy())
            all_masks.append(mask.squeeze(0).numpy())

            if len(all_images) >= shard_size:
                self._save_shard(
                    output_path,
                    shard_idx,
                    all_images,
                    all_masks,
                    windows_params=None,
                    include_features=False,
                )
                shard_idx += 1

                all_images.clear()
                all_masks.clear()

                gc.collect()

        gc.collect()

        if len(all_images) > 0:
            self._save_shard(
                output_path,
                shard_idx,
                all_images,
                all_masks,
                windows_params=None,
                include_features=False,
            )
            all_images.clear()
            all_masks.clear()
            gc.collect()

    def _save_shard(
        self,
        output_path,
        shard_idx,
        all_images,
        all_masks,
        windows_params,
        include_features=True,
    ):

        if include_features:
            features = Features(
                {
                    "image": Array3D(
                        shape=(3, windows_params["ph"], windows_params["pw"]),
                        dtype="float32",
                    ),
                    "mask": Array2D(
                        shape=(windows_params["ph"], windows_params["pw"]),
                        dtype="uint8",
                    ),
                }
            )

        dataset = Dataset.from_dict(
            {
                "image": all_images,
                "mask": all_masks,
            },
            features=features if include_features else None,
        )

        dataset.save_to_disk(f"{output_path}/shard_{shard_idx:03d}")
        del dataset
        gc.collect()
