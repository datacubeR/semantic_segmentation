.PHONY: vaihingen-split, vaihingen-hf, potsdam-hf, loveda-hf, train-segnet-vaihingen-v1, train-unet-vaihingen-v1, train-unetpp-vaihingen-v1, train-upernet-vaihingen-v1, train-segformer-vaihingen-v1, train-swin-vaihingen-v1, train-mask2former-vaihingen-v1

vaihingen-split:
	uv run -m splitters.dataset_splitter --dataset-folder "Vaihingen_dataset" --image-folder "top" --mask-folder "labels" --train-size 0.8 --test-size 0.2

potsdam-split:
	uv run -m splitters.dataset_splitter --dataset-folder "Potsdam_dataset" --image-folder "2_Ortho_RGB" --mask-folder "5_Labels_all" --train-size 0.8 --test-size 0.2

deadtrees-split:
	uv run -m splitters.deadtrees_dataset_splitter --dataset-folder "DeadTrees" --image-folder "dataset_rgb" --mask-folder "dataset_binary" --train-size 0.8 --test-size 0.2

vaihingen-hf: 
	uv run -m src.vaihingen_hf_dataset

potsdam-hf: 
	uv run -m src.potsdam_hf_dataset

loveda-hf: 
	uv run -m src.loveda_hf_dataset

train-segnet-vaihingen-v1:
	uv run -m src.trainer --config config_files/segnet_vaihingen_v1.yaml

train-unet-vaihingen-v1:
	uv run -m src.trainer --config config_files/unet_vaihingen_v1.yaml

train-unetpp-vaihingen-v1:
	uv run -m src.trainer --config config_files/unetpp_vaihingen_v1.yaml

train-upernet-vaihingen-v1:
	uv run -m src.trainer --config config_files/upernet_vaihingen_v1.yaml

train-segformer-vaihingen-v1:
	uv run -m src.trainer --config config_files/segformer_vaihingen_v1.yaml

train-swin-vaihingen-v1:
	uv run -m src.trainer --config config_files/swin_vaihingen_v1.yaml

train-mask2former-vaihingen-v1:
	uv run -m src.trainer --config config_files/mask2former_vaihingen_v1.yaml
