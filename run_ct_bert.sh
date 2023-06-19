#!/bin/bash


for i in 5
do
   python ct-bert2-crossval.py 'ABT_FAMILY' 50 $i
done


