.PHONY: vaihingen-hf, potsdam-hf, loveda-hf, train

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
