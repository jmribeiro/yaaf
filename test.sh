#!/bin/bash

# ############### #
# Setup test venv #
# ############### #

while true; do
    read -p "Do you wish to setup a new virtual environment for testing YAAF? (Y/N) > " answervenv
    case $answervenv in
        [Yy]* ) break ;;
        [Nn]* ) break ;;
        * ) echo "Please answer [y] or [n].";;
    esac
done

if [ "$answervenv" == "y" ]; then
  python -m venv yaaf_test_environment
  source yaaf_test_environment/bin/activate
  pip install --upgrade pip setuptools
  pip install -r requirements.txt
  pip install -e .
fi

while true; do
    read -p "Do you wish to skip supervised learning tests? (slow) (Y/N) > " answerskipslow
    case $answerskipslow in
        [Yy]* ) break ;;
        [Nn]* ) break ;;
        * ) echo "Please answer [y] or [n].";;
    esac
done

# ######### #
# Run tests #
# ######### #

echo "Running framework tests (. = passed, E = Error)"
python -m unittest discover -s tests -t tests

echo "Running pursuit example tests (. = passed, E = Error)"
cp -r examples/pursuit/pretrained-agents .
python -m unittest discover -s examples/pursuit -t examples/pursuit
rm -rf ./pretrained-agents

if [ "$answerskipslow" == "n" ]; then
  echo "Running supervised learning character recognition example tests (slow) (. = passed, E = Error)"
  cp -r examples/character_recognition/ocr_dataset .
  python -m unittest discover -s examples/character_recognition -t examples/character_recognition
  rm -rf ./ocr_dataset
fi

# ####### #
# Cleanup #
# ####### #

if [ "$answervenv" == "y" ]; then
  source deactivate
  rm -rf yaaf_test_environment
fi