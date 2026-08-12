#!/usr/bin/env bash

DATASET="loveda"

MODELS=(
    segnet
    unet
    unetpp
    segformer
    upernet
    swin
    dpt
    deeplab
)

VERSIONS=(
    # 13
    # 14
    15
    16
)

# Combinaciones a omitir: MODEL_VERSION
EXCEPTIONS=(
    "swin_14"
    "swin_16"
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
        # make train DATASET="$DATASET" MODEL="$MODEL" VERSION="$VERSION"
    done
done

echo "All experiments processed."