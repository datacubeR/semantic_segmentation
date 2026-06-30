#!/usr/bin/env bash

configs=(
  segnet_potsdam_v1
  unet_potsdam_v1
  unetpp_potsdam_v1
  segformer_potsdam_v1
  upernet_potsdam_v1
  swin_potsdam_v1
  segnet_potsdam_v2
  unet_potsdam_v2
  unetpp_potsdam_v2
  segformer_potsdam_v2
  upernet_potsdam_v2
)

for cfg in "${configs[@]}"; do
  make train CONFIG_NAME="$cfg"
done

echo "All experiments processed."