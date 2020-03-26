import numpy as np
from scipy.special import expit


class Brain(object):
    def __init__(self, network_size, beta, epoch_of_deducing, drop_rate):

        self.network_size                 = network_size
        self.number_of_layers             = self.network_size.shape[0]

        self.beta                         = beta
        self.epoch_of_deducing            = epoch_of_deducing

        self.drop_rate                    = drop_rate


    def activator(self, x):
        return expit(x)


    def activator_output_to_derivative(self, output):
        return output * (1 - output)


    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        for i in range(self.number_of_layers - 2):

            # apply dropout to hidden layer in the deducing phase.
            binomial              = np.atleast_2d(np.random.binomial(1, 1 - self.drop_rate, size=self.network_size[1 + i]))

            layer                 = self.activator(np.dot(layer_list[-1]                          , self.weight_list[i]                                                          ) * self.slope_list[i] )

            layer                *= binomial

            layer_list.append(layer)

        layer = self.activator(np.dot(layer_list[-1], self.weight_list[-1]) * self.slope_list[-1])

        layer_list.append(layer)

        return   layer_list


    def train_for_input_inner(self,
                       layer_list, desired_output):

        layer_final_error      = desired_output - layer_list[-1]

        layer_delta            = layer_final_error                                                                                              * self.activator_output_to_derivative(layer_list[-1])           * self.slope_list[-1]

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta.dot( self.weight_list[- 1 - i].T                                                          ) )     * self.activator_output_to_derivative(layer_list[- 1 - 1 - i])  * self.slope_list[-1 -1 -i]

        layer_delta        = (layer_delta.dot( self.weight_list[0].T                                                                    ) )     * self.activator_output_to_derivative(layer_list[0])

        self.sudoku_table_inner_batch_update  = layer_delta  * self.beta    * self.sudoku_table_resistor_batch


    def deduce_batch(self, sudoku_table_inner, sudoku_table_resistor, desired_output, weight_list, slope_list):


        self.weight_list = weight_list
        self.slope_list  = slope_list


        # randomly flip, swap and roll the inner values table as well as resistors table for missing and visible numbers without changing their relative
        # positions.
        # ------------------------------------------------
        flip_index   = np.random.randint(3)
        swap_index   = np.random.randint(2)
        roll_index_1 = np.random.randint(3)
        roll_index_2 = np.random.randint(3)

        if flip_index == 0:
            sudoku_table_inner    = sudoku_table_inner
            sudoku_table_resistor = sudoku_table_resistor

        if flip_index == 1:
            sudoku_table_inner    = np.flip(sudoku_table_inner, 0)
            sudoku_table_resistor = np.flip(sudoku_table_resistor, 0)

        if flip_index == 2:
            sudoku_table_inner    = np.flip(sudoku_table_inner, 0)
            sudoku_table_resistor = np.flip(sudoku_table_resistor, 0)
            sudoku_table_inner    = np.flip(sudoku_table_inner, 1)
            sudoku_table_resistor = np.flip(sudoku_table_resistor, 1)

        if swap_index == 1:
            sudoku_table_inner    = np.swapaxes(sudoku_table_inner, 0, 1)
            sudoku_table_resistor = np.swapaxes(sudoku_table_resistor, 0, 1)

        sudoku_table_inner    = np.roll(sudoku_table_inner, 3 * roll_index_1, 0)
        sudoku_table_resistor = np.roll(sudoku_table_resistor, 3 * roll_index_1, 0)
        sudoku_table_inner    = np.roll(sudoku_table_inner, 3 * roll_index_2, 1)
        sudoku_table_resistor = np.roll(sudoku_table_resistor, 3 * roll_index_2, 1)
        #------------------------------------------------

        # generate rows, columns and grids of the inner values table.
        self.sudoku_table_inner_batch    =   np.concatenate((          sudoku_table_inner[:, :].reshape((9, 81)),
                                                           np.swapaxes(sudoku_table_inner[:, :], 0, 1).reshape((9, 81)),
                                                             np.array([sudoku_table_inner[0:3, 0:3].flatten()]),
                                                             np.array([sudoku_table_inner[3:6, 0:3].flatten()]),
                                                             np.array([sudoku_table_inner[6:9, 0:3].flatten()]),
                                                             np.array([sudoku_table_inner[0:3, 3:6].flatten()]),
                                                             np.array([sudoku_table_inner[3:6, 3:6].flatten()]),
                                                             np.array([sudoku_table_inner[6:9, 3:6].flatten()]),
                                                             np.array([sudoku_table_inner[0:3, 6:9].flatten()]),
                                                             np.array([sudoku_table_inner[3:6, 6:9].flatten()]),
                                                             np.array([sudoku_table_inner[6:9, 6:9].flatten()])))

        # generate rows, columns and grids of the resistors table.
        self.sudoku_table_resistor_batch    =   np.concatenate((          sudoku_table_resistor[:, :].reshape((9, 81)),
                                                              np.swapaxes(sudoku_table_resistor[:, :], 0, 1).reshape((9, 81)),
                                                                np.array([sudoku_table_resistor[0:3, 0:3].flatten()]),
                                                                np.array([sudoku_table_resistor[3:6, 0:3].flatten()]),
                                                                np.array([sudoku_table_resistor[6:9, 0:3].flatten()]),
                                                                np.array([sudoku_table_resistor[0:3, 3:6].flatten()]),
                                                                np.array([sudoku_table_resistor[3:6, 3:6].flatten()]),
                                                                np.array([sudoku_table_resistor[6:9, 3:6].flatten()]),
                                                                np.array([sudoku_table_resistor[0:3, 6:9].flatten()]),
                                                                np.array([sudoku_table_resistor[3:6, 6:9].flatten()]),
                                                                np.array([sudoku_table_resistor[6:9, 6:9].flatten()])))

        # deduce and generate inner values update for missing numbers in each row, column and grid by forward-feeding and back-propagation.
        layer_list = self.generate_values_for_each_layer(self.activator( self.sudoku_table_inner_batch))
        self.train_for_input_inner(layer_list, desired_output)

        # apply update to inner values for missing numbers in each row, column and grid.
        sudoku_table_inner[:, :]     += self.sudoku_table_inner_batch_update[0:9].reshape((9, 9, 9))
        sudoku_table_inner[:, :]     += np.swapaxes(self.sudoku_table_inner_batch_update[9:18].reshape((9, 9, 9)), 0, 1)
        sudoku_table_inner[0:3, 0:3] += self.sudoku_table_inner_batch_update[18].reshape((3, 3, 9))
        sudoku_table_inner[3:6, 0:3] += self.sudoku_table_inner_batch_update[19].reshape((3, 3, 9))
        sudoku_table_inner[6:9, 0:3] += self.sudoku_table_inner_batch_update[20].reshape((3, 3, 9))
        sudoku_table_inner[0:3, 3:6] += self.sudoku_table_inner_batch_update[21].reshape((3, 3, 9))
        sudoku_table_inner[3:6, 3:6] += self.sudoku_table_inner_batch_update[22].reshape((3, 3, 9))
        sudoku_table_inner[6:9, 3:6] += self.sudoku_table_inner_batch_update[23].reshape((3, 3, 9))
        sudoku_table_inner[0:3, 6:9] += self.sudoku_table_inner_batch_update[24].reshape((3, 3, 9))
        sudoku_table_inner[3:6, 6:9] += self.sudoku_table_inner_batch_update[25].reshape((3, 3, 9))
        sudoku_table_inner[6:9, 6:9] += self.sudoku_table_inner_batch_update[26].reshape((3, 3, 9))

        # after flipping, swapping and rolling the inner values table, we accordingly return the inner values to their original positions
        # (however the updated value is retained) for next updating.
        # Resistors table needs not be recovered since it is not updated and its original table will be imported in the next epoch.
        # ------------------------------------------------
        sudoku_table_inner    = np.roll(sudoku_table_inner, 9 - 3 * roll_index_2, 1)

        sudoku_table_inner    = np.roll(sudoku_table_inner, 9 - 3 * roll_index_1, 0)

        if swap_index == 1:
            sudoku_table_inner    = np.swapaxes(sudoku_table_inner, 0, 1)

        if flip_index == 0:
            sudoku_table_inner    = sudoku_table_inner

        if flip_index == 1:
            sudoku_table_inner    = np.flip(sudoku_table_inner, 0)

        if flip_index == 2:
            sudoku_table_inner    = np.flip(sudoku_table_inner, 1)
            sudoku_table_inner    = np.flip(sudoku_table_inner, 0)
        # ------------------------------------------------

        return sudoku_table_inner


