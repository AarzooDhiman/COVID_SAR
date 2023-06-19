#!/bin/bash


for i in 1
do
   python ct-bert2-crossval.py 'ABT_FAMILY' 50 $i
done


