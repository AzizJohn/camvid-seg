#!/usr/bin/env bash
# Download the 11-class CamVid dataset (SegNet-Tutorial version) into data/.
# Run on the GPU server from the repo root:  bash scripts/download_camvid.sh
#
# The dataset stays OUTSIDE git (data/ is in .gitignore) - it is downloaded
# directly on every machine that needs it.

set -euo pipefail

mkdir -p data

if [ ! -d data/SegNet-Tutorial ]; then
    echo ">> Cloning SegNet-Tutorial (contains the preprocessed CamVid)..."
    git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial data/SegNet-Tutorial
else
    echo ">> data/SegNet-Tutorial already exists, skipping clone."
fi

# Symlink so the code can always use data/CamVid as the dataset root.
ln -sfn "$(pwd)/data/SegNet-Tutorial/CamVid" data/CamVid

echo ">> Verifying split sizes (expected: train 367, val 101, test 233)..."
status=0
for entry in "train:367" "val:101" "test:233"; do
    split="${entry%%:*}"; expected="${entry##*:}"
    n_img=$(find -L "data/CamVid/${split}" -name '*.png' | wc -l)
    n_ann=$(find -L "data/CamVid/${split}annot" -name '*.png' | wc -l)
    if [ "$n_img" -eq "$expected" ] && [ "$n_ann" -eq "$expected" ]; then
        echo "   ${split}: ${n_img} images / ${n_ann} masks  OK"
    else
        echo "   ${split}: ${n_img} images / ${n_ann} masks  MISMATCH (expected ${expected})"
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo ">> CamVid is ready at data/CamVid"
else
    echo ">> WARNING: unexpected file counts - check the download." >&2
fi
exit "$status"
