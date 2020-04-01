# Deep Deducing on Sudoku

This repository contains codes illustrating how deep deducing solves Sudoku.

## Learning phase content

```
Learning.py            
```

creates set of weight matrix.

```
Brain_for_learning.py
```

where simple deep feedforward neural network for Learning.py is imported.

## Deducing phase content

```
Deducing.py              
```

uses these sets of weight matrix to solve each row, column and grid of the Sudoku table.

```
Brain_for_deducing.py   
```

where simple deep feedforward neural network for Deducing.py is imported.

## Sets of weight matrix

```
self.{6x6}_{original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}_{[1]}_{weight_list}
```

means a single set of weight matrix.


The meaning inside each bracket is listed below:

          {6x6}
          Meaning this neural network is trained to solve each row, column and grid in 6x6 Sudoku.
          
*  {original}
          Meaning the payoff rule in the learning phase for this trained neural network is original.
          In other situation, for example, {odd_even} means the neural network must not only contain total different 
          numbers in a row but also have the odd being separated from even to gain bonus.
          
*  {1}
          Meaning the sample batch in the learning phase for this trained neural network is 1 per each learning epoch.
          
*  {100x100x100}
          Meaning the trained neural network has three hidden layers, each with 100 neurons.
          
*  {30}   Meaning the initial value for the set of slope multiplie to be updated in the learning phase.
          
*  {0.000001}
          Meaning the learning rate in the learning phase for this trained neural network is 0.000001.
*  {200m}
          Meaning the learning epochs in the learning phase for this trained neural network is 200 million or 2*10^8. The learning epochs are usually big in order to force the neural network to over-fit.
*  {[1]}
          Meaning the label of this trained neural network under the training condition {original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}.
          For example, if it is [3], then it means this neural network is the third neural network under the training condition 
          {original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}.
          
*  {weight_list}
          Meaning the set of weight matrix of this trained neural network.

## Prerequisites

```
numpy
```

```
scipy
```

## Running the tests

```
Deducing.py  
```


