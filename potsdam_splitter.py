from split_utils import create_split_folders, split_images_and_masks

IMAGE_GLOB = "Potsdam_dataset/2_Ortho_RGB"
MASK_GLOB = "Potsdam_dataset/5_Labels_all"
TRAIN_FOLDER = "Potsdam_dataset/train"
VAL_FOLDER = "Potsdam_dataset/val"

train_images, val_images, train_masks, val_masks = split_images_and_masks(
    IMAGE_GLOB, MASK_GLOB, test_size=0.2, random_state=42
)


create_split_folders(TRAIN_FOLDER, images=train_images, masks=train_masks)
create_split_folders(VAL_FOLDER, images=val_images, masks=val_masks)
