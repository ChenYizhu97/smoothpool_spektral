#!/bin/bash
for value in FRANKENSTEIN PROTEINS COX2 COX2_MD Mutagenicity ogbg-molclintox ogbg-molbbbp ogbg-molbace
do 
    py main.py --pool smoothpool --lr 0.0005 --dataset $value
    py main.py --pool topkpool --lr 0.0005 --dataset $value
    py main.py --pool sagpool --lr 0.0005 --dataset $value
    py main.py --pool diffpool --lr 0.0005 --dataset $value
    echo $value is evaluated
done

read -p "press any key to exit..."
