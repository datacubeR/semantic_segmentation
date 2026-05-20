.PHONY: vaihingen-hf, potsdam-hf, loveda-hf, trainer

vaihingen-hf: 
	uv run -m src.vaihingen_hf_dataset

potsdam-hf: 
	uv run -m src.potsdam_hf_dataset

loveda-hf: 
	uv run -m src.loveda_hf_dataset

train:
	uv run -m src.trainer