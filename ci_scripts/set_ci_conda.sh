#!/bin/bash

# enter existing conda env.
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

printf "\n============= STEP-0: Build libs =============\n"
if ! pip show pybind11 > /dev/null 2>&1; then
    echo "pybind11 is not installed. Installing..."
    pip install pybind11==2.11.1
fi

git_dir=$(git rev-parse --show-toplevel)
cd $git_dir/loadgen; python setup.py install

if pip show transformers > /dev/null 2>&1; then
    echo "transformers is already installed. Replace transformers to furiosa transformers..."
    pip uninstall -y transformers
    pip install git+https://github.com/furiosa-ai/transformers-comp.git@2b012fcf15006e2cb2b0d9735ebf5b1d08a744a8#egg=transformers
fi

if pip show accelerate > /dev/null 2>&1; then
    echo "accelerate is already installed. Replace accelerate to furiosa accelerate..."
    pip uninstall -y accelerate
    pip install git+https://github.com/furiosa-ai/accelerate-compression.git@4d7b404041834d35727064e5b1dcfcd060319ad6#egg=accelerate
fi

printf "\n============= conda set done! =============\n"
