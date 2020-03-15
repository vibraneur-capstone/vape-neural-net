#!/bin/bash

venv_name="venv"

echo "Initializing virtual environment..."

sleep 1

rm -rf $venv_name

python3 -m venv $venv_name

source $venv_name/bin/activate

printf "\nInstalling dependency \n"
sleep 3

pip3 install -r requirements.txt

printf "\nFinished... \n"

deactivate
