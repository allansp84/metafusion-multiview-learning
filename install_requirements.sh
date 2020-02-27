#!/bin/bash

for req in $(cat requirements.txt); do
    conda install $req -y;
done
