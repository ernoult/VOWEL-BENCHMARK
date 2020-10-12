## Binding events through the mutual synchronization of spintronic nano-neurons

This repository contains the code producing benchmarking the results of [the paper](https://arxiv.org/abs/2001.08044) "Binding events through the mutual synchronization of spintronic nano-neurons" (Romera et al, 2020) against standard deep learning. 


![GitHub Logo](/binding_pic.png)<!-- .element height="20%" width="20%" -->

## Package requirements

Run the following command lines to set the environment using conda:
```
conda create --name EP python=3.6
conda activate EP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch
```


## Commands to reproduce the benchmarking results

+ To visualize any database (for instance R = 10):
  python main.py --action data --R 10

#to run a training simulation
python main.py --action train --epochs 100 --lr 0.05

#to get the csv files of the results for all the vowels, all the noise level, 10 times each
python main.py --action results
