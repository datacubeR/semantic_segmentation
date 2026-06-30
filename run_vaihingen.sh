#!/usr/bin/env bash

configs=(
#   segnet_vaihingen_v1
  unet_vaihingen_v1
#   unetpp_vaihingen_v1
#   segformer_vaihingen_v1
  upernet_vaihingen_v1
#   swin_vaihingen_v1
#   segnet_vaihingen_v2
  unet_vaihingen_v2
#   unetpp_vaihingen_v2
#   segformer_vaihingen_v2
  upernet_vaihingen_v2
)

for cfg in "${configs[@]}"; do
  make train CONFIG_NAME="$cfg"
done

echo "All experiments processed."