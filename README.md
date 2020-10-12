## Binding events through the mutual synchronization of spintronic nano-neurons

This repository contains the code producing benchmarking the results of [the paper](https://arxiv.org/abs/2001.08044) "Binding events through the mutual synchronization of spintronic nano-neurons" (Romera et al, 2020) against standard deep learning.

The project contains the following files:

  + `main.py`: executes the code, with arguments specified in a parser.

  + `utils.py`: contains the classes and functions. 


![GitHub Logo](/binding_pic.PNG)<!-- .element height="20%" width="20%" -->

## Package requirements

Run the following command lines to set the environment using conda:
```
conda create --name EP python=3.6
conda activate EP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch
```

## Details about `main.py`

  `main.py` proceeds in the following way:

  + It first parses arguments typed in the terminal to build a network and get optimization parameters
  (i.e. learning rates, action taken, etc.)

  + It builds the dataset from the .txt files (available upon reasonable request to the authors). 

  + It builds the net using utils.py.

  + It takes one of the three actions that can be fed into the parser.


  The parser takes the following arguments:
  
  |Arguments|Description|Examples|
  |-------|------|------|
  |`action`|Action to be executed in the main.| `--action data `, `--action train `, `--action results`, `--action results-new`|
  |`vowel`|Vowel that is being discriminated (only binary classifications considered). |`--vowel aw`, `--vowel er` `--vowel iy`, `--vowel uw`|
  |`R`| Noise level of the data. | `--R 0`, `--R 10`, `--R 50`, `--R 100`|
  |`lr`| Learning rate used. | `--lr 0.01`|
  |`epochs`| Number of epochs. | `--epochs 100`|
  |`database`| Specifies whether the database is exported in a csv file (Default: False).| `--database`|
  
  main.py can take three different actions:
  
  + `data`: the data is plotted and saved into a csv file if the `database` flag is active.
  
  + `train`: trains the model on the binary classification task for the vowel specified in the parser.
  
  + `results`: gives a statistics of results for binary classification tasks on each vowel (aw, er, iy and uw) for each noise level (R = 0, 10, 50, 100). The model is trained over the whole data available for a given noise level (no test set), on 10 trials. 
  
  + `results-new`: slight variant of `results`, where the model is trained with the noiseless data (R = 0) only and tested against noisy data (R = 10, 50, 100).
  
## Details about `utils.py`

We summarize the different functions of `utils.py` in the following tab.

 |Function|Description|
 |-------|------|
 |`train`| Trains the model for one epoch.|
 |`eval`| Evaluates the model.|
 |`build_dataset`|Builds the dataset for different noise levels.|
 |`plot_data`|Plots the data in a 2-D space.|
 |`plot_results`| Plots the train accuracy as a function of the epochs and the resulting decision boundary in the data space.|


## Commands to reproduce the benchmarking results

+ To visualize any database (for instance R = 10):
  ```
  python main.py --action data --R 10
  ```

+ To run a training simulation (e.g. for the binary classification of the vowel er)
  ```
  python main.py --vowel er --action train --epochs 100 --lr 0.05
  ```

+ To get the csv files of the results for all the vowels, all the noise level, 10 times each (no test set)
  ```
  python main.py --action results
  ```
+ To get the csv files of the results for all the vowels, all the noise level, 10 times each, where the models are trained on noiseless data only and tested against noisy data:
  ```
  python main.py --action results-new
  ```  
  


