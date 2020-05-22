import numpy as np


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Notes for readers
Thank you for reading this note. This code is the learning phase for deep deducing for generating sets of weight matrix to be randomly 
selected in the deducing phase.

You may change or tune any of the following parameters or variables. However, it is recommended that you do so only if the following 
note suggests so. Most of the parameters or variables that you are suggested to change or tune are coherent to supplementary material in the 
paper.

We hope you enjoy it.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part A. Functions for Generating Samples 
This part contains the basic functions for generating samples (input and output datum) for training a deep neural network.
This part contains the basic functions listed below:
    --- generate_one_hot_sudoku_array:
            This function is to generate a 1-D numpy array which represents the concatenation of N one-hotted array representations of N randomly generated numbers. 
            N is the size of each row or column of a Sudoku table. For example, a 6x6 Sudoku has size N = 6. 9x9 Sudoku has size N = 9.
            For example, if 6 numbers:
             [1, 2, 3, 2, 6, 5]
            are chosen, its corresponding output (Sudoku array) will be:
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]].flatten()
             =
             [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    --- return_value:
            This function is to generate a 1-D numpy array which represents the payoff or consequence relating to the Sudoku
            array under the present payoff-rule.
            
            For example, under the original payoff rule, a Sudoku array:
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]].flatten()
            has number "2" over-lapping , and therefore its payoff will be a 1-D numpy array:
             [1, 0, 0, 0, 0, 0]
            Otherwise (for example, all numbers are non-overlapping to each other), its payoff will be an array of ones:
             [1, 1, 1, 1, 1, 1]
             
            If the payoff rule is "odd even", it means the Sudoku array does not only have to have non-overlapping numbers 
            but also have to have these odd numbers being separated from even numbers in order to gain maximum payoff.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def generate_one_hot_sudoku_array(table_size):

    one_hot_sudoku_matrix = np.zeros((table_size, table_size))
    for i in range(one_hot_sudoku_matrix.shape[0]):
        random_number = np.random.randint(table_size)
        one_hot_sudoku_matrix[i][random_number] = 1

    return one_hot_sudoku_matrix.flatten()



def return_value(one_hot_sudoku_array, table_size, payoff_rule):
    one_hot_sudoku_matrix = one_hot_sudoku_array.reshape((table_size, table_size))

    if payoff_rule == "original":

        if np.count_nonzero(one_hot_sudoku_matrix.sum(axis=0) == 0) >=1 :
            returned = np.zeros(table_size)
            returned[0] = 1
            return returned

    if payoff_rule == "odd even":
        for j in range(one_hot_sudoku_matrix.shape[1]):
            if np.count_nonzero(one_hot_sudoku_matrix[:, j] == 1) != 1:
                returned        = np.zeros(table_size)
                returned[0:1] = 1
                return returned
        for i in range(one_hot_sudoku_matrix.shape[0] - 1):
            if ((np.argmax(one_hot_sudoku_matrix[i]) + 1) + (np.argmax(one_hot_sudoku_matrix[i+1]) + 1)) % 2 == 0:
                returned = np.zeros(table_size)
                returned[0:1] = 1
                return returned

    returned = np.ones(table_size)

    return returned

# parameter referring to the size of row or column of the Sudoku table we are trying to solve. We recommend readers to try different numbers.
table_size                = 6
# parameter referring to the payoff rule. We recommend readers to try "odd even".
payoff_rule               = "original"
# parameter referring to the batch size of the samples used to train the neural network. We recommend readers to try different batch size.
batch_size                = 1


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part B. Initializing Set of Weight Matrix and Importing Model
In this part, we import the model for learning and create an object or instance from the imported class.
We define:
--- network_size:
    The topology of the deep neural network. For example, if it is [36, 100, 100, 100, 6], it means the deep neural network
    has one input layer with 36 neurons, three hidden layers each with 100 neurons, and an output layer with 6 neurons.
--- alpha:
    The learning rate for the set of weight matrix and slope multiplier.
--- epoch_of_learning:
    Learning epochs under which traditional SGD is performed in every epoch upon the set of weight matrix and slope multiplier.
--- Machine:
    The name of the object or instance created from the class "Brain".
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


from Brain_for_learning import *
# this parameter refers to the topology of the neural network. We recommend readers to try different numbers.
network_size              = np.array([table_size * table_size, 100, 100, 100, table_size])
# this parameter refers to intial slopes for the activation/sigmoid functions in the hidden and output layers of the neural network. We recommend readers to try different numbers.
slope                     = 30
# this parameter refers to learning rate. We recommend readers to try different numbers.
alpha                     = 0.000001
# this parameter refers to learning epochs. We recommend readers to try different numbers.
epoch_of_learning         = 100000000

Machine                   = Brain(network_size, slope, alpha, epoch_of_learning)


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part C. Generating Samples and Training by Model
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# this parameter decides whether the program will train weight matrix upon existing weight matrix. We recommend readers to try different numbers.
retrain = False

if retrain == True:

    Machine.weight_list   = np.load("self.6x6_original_1_100x100x100_30_0.000001_100m_[10]_weight_list.npy" , allow_pickle=True)
    Machine.slope_list    = np.load("self.6x6_original_1_100x100x100_30_0.000001_100m_[10]_slope_list.npy"  , allow_pickle=True)


for i in range(epoch_of_learning):
    print(i)
    input_list  = list()
    output_list = list()
    for j in range(batch_size):
        one_hot_sudoku_array       = generate_one_hot_sudoku_array(table_size)
        one_hot_sudoku_array_value = return_value(one_hot_sudoku_array, table_size, payoff_rule)
        input_list.append(one_hot_sudoku_array)
        output_list.append(one_hot_sudoku_array_value)
    input_list  = np.asarray(input_list)
    output_list = np.asarray(output_list)
    Machine.learn_batch(input_list, output_list)


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part D. Saving the Trained Set of Weight Matrix for MWM-SGD
This part simply saves the trained set of weight matrix (as well as their corresponding sets of slope multiplier).
The meaning of the notations are listed below:
    --- self.{6x6}_{original}_{1}_{100x100x100}_{30}_{0.000001}_{100m}_{[1]}_{weight_list}
            The meaning of each bracket is listed below:
                --- {6x6}
                        Meaning this neural network is trained to solve each row, column and grid in 6x6 Sudoku.
                --- {original}
                        Meaning the payoff rule in the learning phase for this trained neural network is original.
                        In other situation, for example, {odd_even} means the neural network must not only contain total different 
                        numbers in a row but also have the odd being separated from even to gain payoff/bonus.
                --- {1}
                        Meaning the sample batch in the learning phase for this trained neural network is 1 per each learning epoch.
                --- {100x100x100}
                        Meaning the trained neural network has three hidden layers, each with 100 neurons.
                --- {30}
                        Meaning the initial value for the set of slope multiplier to be updated in the learning phase.
                --- {0.000001}
                        Meaning the learning rate in the learning phase for this trained neural network is 0.000001.
                --- {100m}
                        Meaning the learning epochs in the learning phase for this trained neural network is 100 million or 10^8. The learning epochs 
                        are usually big in order to force the neural network to over-fit.
                --- {[1]}
                        Meaning the label of this trained neural network under the training condition {original}_{1}_{100x100x100}_{30}_{0.000001}_{100m}.
                --- {weight_list}
                        Meaning the set of weight matrix of this trained neural network.
            Each bracket correlates to the notations in supplementary material in the paper.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# this two lines save the trained set of weight matrix later to be used/selected in the dedcuing phase. We recommend readers to try different numbers.
np.save("self.6x6_original_1_100x100x100_30_0.000001_100m_[10]_weight_list"             , Machine.weight_list        )
np.save("self.6x6_original_1_100x100x100_30_0.000001_100m_[10]_slope_list"              , Machine.slope_list         )




