#!/bin/bash
#This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).


for i in 5
do
   python ct-bert2-crossval.py 'ABT_FAMILY' 50 $i
   #python ct-bert2-crossval.py 'AUTHOR_OR' 50 $i
   #python ct-bert2-crossval.py 'FAMILY_OR' 50 $i
done


