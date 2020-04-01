# Deep Deducing on Sudoku

This repository contains codes illustrating how deep deducing solves Sudoku.

## Content

Its content is as follow:

### Learning phase content

```
Learning.py              --- creates set of weight matrix.
```

```
Brain_for_learning.py    --- imports simple deep feedforward neural network for Learning.py.
```

### Deducing phase content

```
Deducing.py              --- uses these sets of weight matrix to solve each row, column and grid of the Sudoku table.
```

```
Brain_for_deducing.py    --- imports simple deep feedforward neural network for Deducing.py.
```

### Sets of weight matrix

```
self.6x6_original_1_100x100x100_30_0.000001_100m_[1]_slope_list --- a single set of weight matrix.
```

## Prerequisites

```
numpy
```

## Running the tests

Simply run any of the following content:

```
Deducing.py  
```


