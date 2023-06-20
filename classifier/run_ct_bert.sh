#!/bin/bash


for i in 5
do
   python ct-bert2-crossval.py 'ABT_FAMILY' 50 $i
   #python ct-bert2-crossval.py 'AUTHOR_OR' 50 $i
   #python ct-bert2-crossval.py 'FAMILY_OR' 50 $i
done


