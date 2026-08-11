#!/usr/bin/env bash

DATASET="potsdam"

MODELS=(
    # segnet
    # unet
    # unetpp
    # segformer
    # upernet
    # swin
    dpt
    deeplab
)

VERSIONS=(
    1
    2
)

# Combinaciones a omitir: MODEL_VERSION
EXCEPTIONS=(
    "swin_14"
    # "segnet_13"
    # "dpt_13"
    # "segformer_14"
)

for VERSION in "${VERSIONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        COMBINATION="${MODEL}_${VERSION}"

        # Skip if the combination is in the exceptions list
        if [[ " ${EXCEPTIONS[*]} " =~ " ${COMBINATION} " ]]; then
            echo "Skipping ${MODEL}_${DATASET}_v${VERSION}"
            continue
        fi

        echo "Running ${MODEL}_${DATASET}_v${VERSION}"
        make train DATASET="$DATASET" MODEL="$MODEL" VERSION="$VERSION"
    done
done

echo "All experiments processed."