#!/bin/bash
for value in FRANKENSTEIN ogbg-molclintox PROTEINS COX2 COX2_MD Mutagenicity  ogbg-molbbbp ogbg-molbace
do 
    py -u main.py --pool smoothpool --dataset $value
    py -u main.py --pool topkpool --dataset $value
    py -u main.py --pool sagpool --dataset $value
    py -u main.py --pool diffpool --dataset $value
    echo $value is evaluated
done

read -p "press any key to exit..."
