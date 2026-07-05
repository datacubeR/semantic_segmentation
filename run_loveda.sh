#!/usr/bin/env bash

configs=(
  segnet_loveda_v13
  unet_loveda_v13
  unetpp_loveda_v13
  segformer_loveda_v13
  upernet_loveda_v13
  swin_loveda_v13
  dpt_loveda_v13
  segnet_loveda_v14
  unet_loveda_v14
  unetpp_loveda_v14
  segformer_loveda_v14
  upernet_loveda_v14
  dpt_loveda_v14

)

for cfg in "${configs[@]}"; do
  make train CONFIG_NAME="$cfg"
done

echo "All experiments processed."