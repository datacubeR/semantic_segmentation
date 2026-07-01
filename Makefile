.PHONY: vaihingen-split, potsdam-split, loveda-split, deadtrees-split,train

#  train-segnet-vaihingen-v1, train-unet-vaihingen-v1, train-unetpp-vaihingen-v1, train-upernet-vaihingen-v1, train-segformer-vaihingen-v1, train-swin-vaihingen-v1, train-mask2former-vaihingen-v1

PATCH_SIZE ?= 256
NAME ?= 
VT ?= 0
TR ?= 1
TRAIN_SHARDS ?= 5000
VAL_SHARDS ?= 10
TEST_SHARDS ?= 10
VERSION ?= 
DATASET ?= 
MODEL ?= 

vaihingen-split:
	uv run -m splitters.dataset_splitter --dataset-folder "Vaihingen_dataset" --image-folder "top" --mask-folder "labels" --train-size 0.8 --test-size 0.2

potsdam-split:
	uv run -m splitters.dataset_splitter --dataset-folder "Potsdam_dataset" --image-folder "2_Ortho_RGB" --mask-folder "5_Labels_all" --train-size 0.8 --test-size 0.2

loveda-split:
	uv run -m splitters.loveda_dataset_splitter --dataset-folder LoveDA

deadtrees-split:
	uv run -m splitters.deadtrees_dataset_splitter --dataset-folder "DeadTrees" --image-folder "dataset_rgb" --mask-folder "dataset_binary" --train-size 0.8 --test-size 0.2

hf:
	uv run -m src.create_hf_dataset \
		--dataset-folder $(NAME) \
		--patch-size $(PATCH_SIZE) \
		--train-shard-size $(TRAIN_SHARDS) \
		--val-shard-size $(VAL_SHARDS) \
		--test-shard-size $(TEST_SHARDS) \
		$(if $(filter 1,$(TR)),-tr) \
		$(if $(filter 1,$(VT)),-vt)

# loveda-hf: 
# 	uv run -m src.loveda_hf_dataset

train:
	@if [ "$(DATASET)" = "loveda" ] || [ "$(DATASET)" = "deadtrees" ]; then \
		uv run -m src.simpletrainer --model $(MODEL) --dataset $(DATASET) --version $(VERSION); \
	else \
		uv run -m src.gridtrainer --model $(MODEL) --dataset $(DATASET) --version $(VERSION); \
	fi



# train-segnet-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/segnet_vaihingen_v1.yaml

# train-unet-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/unet_vaihingen_v1.yaml

# train-unetpp-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/unetpp_vaihingen_v1.yaml

# train-upernet-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/upernet_vaihingen_v1.yaml

# train-segformer-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/segformer_vaihingen_v1.yaml

# train-swin-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/swin_vaihingen_v1.yaml

# train-mask2former-vaihingen-v1:
# 	uv run -m src.trainer --config config_files/mask2former_vaihingen_v1.yaml
