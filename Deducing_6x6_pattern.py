import numpy as np
from scipy.special import expit
import time


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Notes for readers
Thank you for reading this note. This code is the deducing phase for deep deducing for leveraging sets of weight matrix to perform MWM-
SGD to solve 6x6 Sudoku table with pattern grids. If you want to have some fun, you may change the topology of the grids by 
manually change the grid indexes in the following functions or class functions:
    --- Part A. 
        --- generate_answer_table
        --- last_check  
    --- Part B-3.
        --- from Brain_for_deducing_6x6_pattern import * 
            --- deduce_batch

This code is the same as "Deducing_6x6" and "Deducing_6x6_pattern" except their functions parameters (which we don't recommend readers to change)
or imports listed below are different:
    --- Part A. 
        --- generate_answer_table
        --- last_check  
    --- Part B.
        --- table_size
    --- Part B-3.
        --- from Brain_for_deducing_6x6_pattern import * 
We disected the same code into three codes for convenience reasons. So all you have to do is to change the parameters which the following
note suggests you to change.

You may change or tune any of the following parameters or variables. However, it is recommended that you do so only if the following 
note suggests so. Most of the parameters or variables that you are suggested to change or tune are coherent to Appendix A and C in the 
paper.

We hope you enjoy it.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part A. Funtions for Generating Sudoku Table, etc. 
This part contains the basic functions listed below:
    --- generate_asnwer_table:
            This function is to generate a (6, 6) numpy array which represents an answer table, where numbers will later 
            be dropped uniformly/randomly to generate a Sudoku table or a puzzle for machine to solve.
    --- generate_Sudoku_table:
            This function is to generate a (6, 6) numpy array which represents a Sudoku table by uniformly/randomly dropping
            visible numbers in the answer table.
            Missing numbers will be represented as 0s.
    --- return_inner:
            This function is to generate a (6, 6, 6) numpy array which represents a inner values table for the generated Sudoku table.
            For an extremely concise illustration, a (2, 2) numpy array:
                [[1, 2],
                 [0, 0]] 
            will be represented as a (2, 2, 2) numpy array (with noise):
                [[[ 3.51, -3.54], [-3.50,  3.51]],
                 [[-3.52, -3.53], [-3.51, -3.52]]]
            The starting inner values may not have to be -3.5. However, to match the records in Appendix C in the paper. It is recommended 
            it is set to be -3.5.
    --- return_resistor:
            This function is to generate a (6, 6, 6) numpy array which tells the program "not" to update inner values for visible 
            numbers in the deducing epcohs.
            For an extremely concise illustration, a (2, 2) numpy array:
                [[1, 2],
                 [0, 0]] 
            will generate a resistors table which is a (2, 2, 2) numpy array:
                [[[0, 0], [0, 0]],
                 [[1, 1], [1, 1]]]
            so the inner values for algebra/int 1 and 2 will be kept from updated in the deducing epochs.
    --- mandatory_pulse:
            This function is to generate two (6, 6, 6) numpy arrays which separately represent the newest inner values table
            and the resistors table by finding the amax of inner values of missing numbers, moving the specific missing number
            to visible number, and re-initializing the inner values for the rest of the missing numbers.
            For an extremely concise illustration, after certain deducing epochs, the inner values table and its corresponding resistors table:
                [[[ 3.51, -3.54], [-3.50,  3.51]],
                 [[-1.52,  2.58], [-3.57, -4.59]]]
                [[[0, 0], [0, 0]],
                 [[1, 1], [1, 1]]]
            will be separately replaced by their new status:
                [[[ 3.51, -3.54], [-3.50,  3.51]],
                 [[-3.56,  3.58], [-3.54, -3.52]]]
                [[[0, 0], [0, 0]],
                 [[0, 0], [1, 1]]]
            As we can see, the missing number s^{1, 0} has the highest digit 2.58 among -1.52, -3,57 and -4,59 (in all inner values
            for the missing numbers).
            Therefore, the corresponding inner value and resistor value of the missing number s^{1, 0} is updated,
            which also means that s^{1, 0} is updated as a visible number.
            The inner values for the rest of the missing numbers (e.g. s^{1, 1}) will also be re-initialized.
    --- last_check:
            This function simply checks if the machine successfully solves a Sudoku table according to the set rules. In this code, it checks
            if there are any number overlapping in the rows, columns or grids. If there is, then it returns 0, which means 0 score.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def generate_answer_table(row, column):

    answer_table = np.zeros((row, column))

    for i in range(row):
        for j in range(column):
            random_number      = np.random.randint(row) + 1
            grid_1 = np.array([answer_table[0, 0], answer_table[0, 1], answer_table[1, 1],
                               answer_table[2, 0], answer_table[2, 1], answer_table[3, 0]])
            grid_2 = np.array([answer_table[3, 1], answer_table[3, 2], answer_table[4, 0],
                               answer_table[4, 1], answer_table[5, 0], answer_table[5, 1]])
            grid_3 = np.array([answer_table[0, 1], answer_table[0, 2], answer_table[0, 3],
                               answer_table[1, 2], answer_table[1, 3], answer_table[2, 2]])
            grid_4 = np.array([answer_table[3, 3], answer_table[4, 2], answer_table[4, 3],
                               answer_table[5, 2], answer_table[5, 3], answer_table[5, 4]])
            grid_5 = np.array([answer_table[0, 4], answer_table[0, 5], answer_table[1, 4],
                               answer_table[1, 5], answer_table[2, 3], answer_table[2, 4]])
            grid_6 = np.array([answer_table[2, 5], answer_table[3, 4], answer_table[3, 5],
                               answer_table[4, 4], answer_table[4, 5], answer_table[5, 5]])
            fail = 0
            if ((i==0)&(j==0)) | ((i==0)&(j==1)) | ((i==1)&(j==1)) | \
               ((i==2)&(j==0)) | ((i==2)&(j==1)) | ((i==3)&(j==0)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_1)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            if ((i==3)&(j==1)) | ((i==3)&(j==2)) | ((i==4)&(j==0)) | \
               ((i==4)&(j==1)) | ((i==5)&(j==0)) | ((i==5)&(j==1)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_2)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            if ((i==0)&(j==1)) | ((i==0)&(j==2)) | ((i==0)&(j==3)) | \
               ((i==1)&(j==2)) | ((i==1)&(j==3)) | ((i==2)&(j==2)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_3)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            if ((i==3)&(j==3)) | ((i==4)&(j==2)) | ((i==4)&(j==3)) | \
               ((i==5)&(j==2)) | ((i==5)&(j==3)) | ((i==5)&(j==4)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_4)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            if ((i==0)&(j==4)) | ((i==0)&(j==5)) | ((i==1)&(j==4)) | \
               ((i==1)&(j==5)) | ((i==2)&(j==3)) | ((i==2)&(j==4)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_5)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            if ((i==2)&(j==5)) | ((i==3)&(j==4)) | ((i==3)&(j==5)) | \
               ((i==4)&(j==4)) | ((i==4)&(j==5)) | ((i==5)&(j==5)) :
                while ((random_number in answer_table[i, :]) ) | ((random_number in answer_table[:, j]) ) | (random_number in grid_6)   :
                    random_number = np.random.randint(row) + 1
                    fail += 1
                    if fail >= 2000:
                        return generate_answer_table(row, column)
            answer_table[i][j] = random_number

    return answer_table


def generate_sudoku_table(answer_table, numbers_missing):
    size          = answer_table.shape[0]
    sudoku_table = answer_table.flatten()

    # This iteration randomly sets numbers to 0 in an answer table
    for i in range(numbers_missing):
        random_index = np.random.randint(sudoku_table.shape[0])
        while sudoku_table[random_index] == 0:
            random_index = np.random.randint(sudoku_table.shape[0])
        sudoku_table[random_index] = 0

    sudoku_table = sudoku_table.reshape((size, size))

    return sudoku_table


def return_inner(sudoku_table, inner):
    shape = sudoku_table.shape[0]
    sudoku_table_inner = np.zeros((shape ,shape , shape ))

    for i in range(sudoku_table.shape[0]):
        for j in range(sudoku_table.shape[1]):
            if sudoku_table[i][j] == 0:
                array = (np.random.random(shape) - 0.5) * 0.1  + inner
                sudoku_table_inner[i][j] = array
            else:
                array = (np.random.random(shape) - 0.5) * 0.1  + inner
                np.put(array, sudoku_table[i][j] - 1, (np.random.random(shape) - 0.5) * 0.1  - inner)
                sudoku_table_inner[i][j] = array

    return sudoku_table_inner


def return_resistor(sudoku_table):
    shape = sudoku_table.shape[0]
    sudoku_table_resistor = np.zeros((shape ,shape , shape ))

    for i in range(sudoku_table.shape[0]):
        for j in range(sudoku_table.shape[1]):
            if sudoku_table[i][j] == 0:
                array = np.zeros(shape) + 1
                sudoku_table_resistor[i][j] = array
            else:
                array = np.zeros(shape)
                sudoku_table_resistor[i][j] = array

    return sudoku_table_resistor


def mandatory_pulse(sudoku_table_inner, sudoku_table_resistor, inner):

    i, j, k = np.where(sudoku_table_inner + sudoku_table_resistor * 10000000 == np.amax(sudoku_table_inner + sudoku_table_resistor * 10000000))
    i = i[0]
    j = j[0]
    k = k[0]

    sudoku_table_inner[i, j]    = (np.random.random((sudoku_table_inner[i, j].shape[0],))-0.5) * 0.1 + inner
    sudoku_table_inner[i, j, k] = (np.random.random((1))-0.5)                                  * 0.1 - inner

    sudoku_table_resistor[i, j] = sudoku_table_resistor[i, j] * 0

    for m in range(sudoku_table_inner.shape[0]):
       for n in range(sudoku_table_inner.shape[1]):
           if (sudoku_table_resistor[m, n][0] == 1) & ((m != i) | (n != j)) :
               sudoku_table_inner[m, n] = (np.random.random((sudoku_table_inner[i, j].shape[0],))-0.5) * 0.1  + inner

    return sudoku_table_inner, sudoku_table_resistor


def last_check(machine_table):
    for i in range(machine_table.shape[0]):
        for j in range(machine_table.shape[1]):
            target_number = machine_table[i][j]
            if (np.count_nonzero(machine_table[i, :] == target_number) != 1) |  (np.count_nonzero(machine_table[:, j] == target_number) != 1)  :
                return 0

    examined_grid = np.array([machine_table[0,0], machine_table[1,0], machine_table[1,1],
                              machine_table[2,0], machine_table[2,1], machine_table[3,0]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    examined_grid = np.array([machine_table[3,1], machine_table[3,2], machine_table[4,0],
                              machine_table[4,1], machine_table[5,0], machine_table[5,1]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    examined_grid = np.array([machine_table[0,1], machine_table[0,2], machine_table[0,3],
                              machine_table[1,2], machine_table[1,3], machine_table[2,2]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    examined_grid = np.array([machine_table[3,3], machine_table[4,2], machine_table[4,3],
                              machine_table[5,2], machine_table[5,3], machine_table[5,4]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    examined_grid = np.array([machine_table[0,4], machine_table[0,5], machine_table[1,4],
                              machine_table[1,5], machine_table[2,3], machine_table[2,4]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    examined_grid = np.array([machine_table[2, 5], machine_table[3, 4], machine_table[3, 5],
                              machine_table[4, 4], machine_table[4, 5], machine_table[5, 5]])
    for i in range(examined_grid.shape[0]):
        target_number = examined_grid[i]
        if (np.count_nonzero(examined_grid[:] == target_number) != 1):
            return 0

    return 1


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part B. Starting Trials
This part is the start of test. Here we define:
    --- times_of_trial:
            How many times will the test be run. If it is 100, it means there wll be 100 Sudoku tables waiting for deep deducing to solve.
    --- score:
            The starting score for deep deducing. Of course, it is 0.
    --- repetition_checker:
            A simple list to record all the final table proposed by deep deducing to check if there is any solved table repeated.
            It is for testing the variability of deep deducing under blank Sudoku with all the numbers being dropped.
    --- repetition_timer:
            To record how many times have deep deducing repeatedly proposed the same table.
            It is for testing the variability of deep deducing under blank Sudoku with all the numbers being dropped.
    --- table_size:
            The size of the row, column of the Sudoku table to be solved.
    --- numbers_missing:
            It determines how many numbers will be droppped in a Sudoku table. 
            For example, for 6x6 Sudoku table, if it is 18, it means there will be 18 numbers missing, lefting only 18 numbers (18 given).
            If it is 36, then it is a blank Sudoku table (0 given).
    --- answer_table:
            A 2-D numpy array.
            It is the answer table that have non-overlapping numbers in every row, column or grids.
    --- sudoku_table:
            A 2-D numpy array.
            It is the answer table with numbers missing (the amount of which equals to numbers_missing).
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Referring to times of test. we recommend readers to change this parameter.
times_of_trial      = 1000

score               = 0

repetition_checker  = []

repetition_timer    = 0

for table in range(times_of_trial):

    table_size      = 6
    # Referring to the amount of numbers being dropped in an answer table. we recommend readers to change this parameter, at most 36 for 6x6 Sudoku.
    numbers_missing = 36

    answer_table    = generate_answer_table(table_size, table_size)

    sudoku_table    = generate_sudoku_table(answer_table, numbers_missing)


    """
      -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                 Part B-1. Loading the Trained Sets of Weight Matrix for MWM-SGD
      This part simply loads sets of weight matrix (as well as their corresponding sets of slope multiplier trained by deep learning in the 
      learning phase for MWM-SGD.
      The meaning of the notations are listed below:
          --- weight_list
                  Refers to the set of weight matrix trained by deep learning in the learning phase.
          --- slope_list
                  Refers to the set of slope multiplier trained by deep learning in the learning phase.
          --- weight_lists
                  Refers to the sets of weight matrix to be randomly selected/sampled in the deducing epochs to perform MWM-SGD.
          --- slope_lists
                  Refers to the sets of slope multiplier to be randomly selected/sampled in the deducing epochs to perform MWM-SGD.
          --- self.{6x6}_{original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}_{[1]}_{weight_list}
                  The meaning of each bracket is listed below:
                      --- {6x6}
                              Meaning this neural network is trained to solve each row, column and grid in 6x6 Sudoku.
                      --- {original}
                              Meaning the payoff rule in the learning phase for this trained neural network is original.
                              In other situation, for example, {odd_even} means the neural network must not only contain total different 
                              numbers in a row but also have the odd being separated from even to gain bonus.
                      --- {1}
                              Meaning the sample batch in the learning phase for this trained neural network is 1 per each learning epoch.
                      --- {100x100x100}
                              Meaning the trained neural network has three hidden layers, each with 100 neurons.
                      --- {30}
                              Meaning the initial value for the set of slope multiplier to be updated in the learning phase.
                      --- {0.000001}
                              Meaning the learning rate in the learning phase for this trained neural network is 0.000001.
                      --- {200m}
                              Meaning the learning epochs in the learning phase for this trained neural network is 200 million or 2*10^8. The learning epochs 
                              are usually big in order to force the neural network to over-fit.
                      --- {[1]}
                              Meaning the label of this trained neural network under the training condition {original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}.
                              For example, if it is [3], then it means this neural network is the third neural network under the training condition 
                              {original}_{1}_{100x100x100}_{30}_{0.000001}_{200m}.
                      --- {weight_list}
                              Meaning the set of weight matrix of this trained neural network.
                  Each bracket correlates to the notations in Appendix C in the paper.
      -----------------------------------------------------------------------------------------------------------------------------------------------------
      """

    # we recommend readers to try different size of sets of weight matrix by deleting some of the weight or slope list to grasp a feeling how
    # size of sets of weight matrix affect overall accuracy and variability.
    weight_lists = list()
    slope_lists  = list()

    weight_list        = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[1]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[1]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[2]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[2]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[3]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[3]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[4]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[4]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[5]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.6x6_original_1_100x100x100_30_0.000001_200m_[5]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)


    """
      -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                 Part B-2. Initializing Inner, etc.
      This part initializes inner values for missing and visible numbers (as well as their resistors to tell program to update only inner values
      for missing numbers) and desired output (array of ones).
      The definition are as follow:
      --- inner:
            The starting inner values for missing and visible numbers. For example, if the starting inner value is -3.5, then the inner value
            for the corresponding missing number in 6x6 Sudoku will be:
            [-3.51, -3.52, -3.49, -3.51, -3.53, -3.48]     (with uniformly generated noise)
            And the inner value for visible number "2" will be:
            [-3.50, -3.51, -3.48, -3.52, -3.53, -3.49]     (with uniformly generated noise)
            Why -3.5? Because 1/(1+e^(-3.5)) = 0.029... is close to 0. So the initial neurons will have zero energy before mandatory pulse takes place.
            The starting inner values may not have to be -3.5. However, to match the records in Appendix C in the paper. It is recommended 
            it is set to be -3.5.
      --- sudoku_table_inner:
            It returns the inner values table for a Sudoku table.
      --- sudoku_table_resistor:
            It returns the resistors table for a Sudoku table.
      --- desired_output:
            The desired output datum. For example, to solve a 6x6 Sudoku table, the rows, columns and grids of the inner values table will form
            18 generated output datum when fed-forward. Then we will need corresponding 18 desired output datum to tell the machine to update the
            inner values for the missing numbers accordingly by using back-prop. So it will be a 2-D (18, 6) numpy array with ones.
      -----------------------------------------------------------------------------------------------------------------------------------------------------
      """


    inner                  = -3.500
    sudoku_table_inner     = return_inner(sudoku_table, inner)
    sudoku_table_resistor  = return_resistor(sudoku_table)

    desired_output         = np.ones((table_size * 3, table_size))


    """
      -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                 Part B-3. Importing Model
      In this part, we import the model for deducing and create an object or instance from the imported class.
      We define:
      --- network_size:
            The topology of the deep neural network. For example, if it is [36, 100, 100, 100, 6], it means the deep neural network
            has one input layer with 36 neurons, three hidden layers each with 100 neurons, and an output layer with 6 neurons.
      --- beta:
            The deducing rate for the inner values for the missing numbers.
      --- epoch_of_deducing:
            Deducing epochs under which MWM-SGD is performed in every epoch upon inner values for the missing numbers.
      --- Machine:
            The name of the object or instance created from the class "Brain".
      -----------------------------------------------------------------------------------------------------------------------------------------------------
      """


    from Brain_for_deducing_6x6_pattern import *

    network_size                = np.array([table_size * table_size, 100, 100, 100, table_size])
    # Referring to deducing rate. we recommend readers to change this parameter.
    beta                        = 0.1
    # Referring to deducing epcohs "between each mandatory pulse". we recommend readers to change this parameter.
    epoch_of_deducing           = 2000
    # Referring to the rate of neurons dropped out in the hidden layers. For example, if drop_rate = 0.2, it means 20% of the neurons in the hidden layers will be dropped out on a random base.  we recommend readers to change this parameter.
    drop_rate                   = 0.1

    Machine                     = Brain(network_size, beta, epoch_of_deducing, drop_rate)


    """
      -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                 Part B-4. Selecting Set of Weight Matrix and Deducing by Model
      -----------------------------------------------------------------------------------------------------------------------------------------------------
      """


    print("---------------The answer table------------------")
    print(answer_table)
    print("---------------The Sudoku table------------------")
    print(sudoku_table)

    for times in range(numbers_missing):


        start = time.time()


        """
            -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                       Part B-4-1. Back-propagation and WMW-SGD
            In this part, under each deducing epoch, we randomly select a set of trained weight matrix and its corresponding set of slope multiplier 
            every time before the inner values for the missing numbers is updated by forward-feeding and back-propagation.
            This part is the very core of deep deducing.
            -----------------------------------------------------------------------------------------------------------------------------------------------------
            """
        for i in range(epoch_of_deducing):


            random_index         = np.random.randint(np.asarray(weight_lists).shape[0])
            weight_list          = weight_lists[random_index]
            slope_list           = slope_lists[random_index]


            sudoku_table_inner   = Machine.deduce_batch(sudoku_table_inner, sudoku_table_resistor, desired_output, weight_list, slope_list)


        print("---------------The output of the present inner values of the missing and visible numbers------------------")
        print(np.round(expit(sudoku_table_inner), 2))


        """
            -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                       Part B-4-2. Mandatory Pulse
            In this part, after certain deducing epochs, the program finds the amax of the inner values for the missing numbers.
            Whichever missing number that holds the highest specific inner value will be delegated as the new visible number (by 
            re-allocating its inner value and resistor), and the inner values of the rest of the missing numbers will be re-initialized.
            -----------------------------------------------------------------------------------------------------------------------------------------------------
            """
        sudoku_table_inner, sudoku_table_resistor   = mandatory_pulse(sudoku_table_inner, sudoku_table_resistor, inner)


        print("---------------The present numbers solved by machine--------------------")
        machine_table = np.zeros((sudoku_table_inner.shape[0], sudoku_table_inner.shape[1]))
        for i in range(sudoku_table_inner.shape[0]):
            for j in range(sudoku_table_inner.shape[1]):
                if (sudoku_table_resistor[i][j][0] == 0):
                    machine_table[i][j] = np.argmax(sudoku_table_inner[i][j]) + 1
        print(machine_table)

        print("---------------Times of mandatory pulse so far----------------------------")
        print(times + 1)

        print("---------------Elasped Time-----------------------------------------------")
        end = time.time()
        print(end - start)

        if np.count_nonzero(sudoku_table_resistor) == 0:
            break


    """
      -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                                 Part B-5. Printing out Result of the Final Table Proposed by the Machine, etc.
      In this part, we simply print out all relative results.
      -----------------------------------------------------------------------------------------------------------------------------------------------------
      """
    print("---------------Final comparison------------------------------------------------------------------------------------------")

    print("---------------The answer table-------------------------------------------------------------------------------------------")
    print(answer_table)

    print("---------------The Sudoku table------------------------------------------------------------------------------------------")
    print(sudoku_table)

    print("---------------The final table proposed by the machine-----------------------------------------------------------------")
    machine_table = sudoku_table
    for i in range(sudoku_table.shape[0]):
        for j in range(sudoku_table.shape[1]):
            machine_table[i][j] = np.argmax(sudoku_table_inner[i][j]) + 1
    print(machine_table)

    print("---------------The difference between the final table proposed by the machine and the answer table-----------------")
    print(answer_table - machine_table)

    print('--------------------------------------------------------------------------------------------------------------------------------Table: ', table + 1)
    if last_check(machine_table) == 1:
        score += 1
    print('--------------------------------------------------------------------------------------------------------------------------------Present score: ', score)
    if any(np.array_equal(machine_table , i) for i in np.asarray(repetition_checker)) == True:
        repetition_timer += 1
    repetition_checker.append(machine_table)
    print('--------------------------------------------------------------------------------------------------------------------------------Repetition: ', repetition_timer)


print("Times of trial:")
print(times_of_trial)
print("Total score:")
print(score)
print("Repetition:")
print(repetition_timer)

