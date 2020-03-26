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

        # generate rows, columns and grids of the inner values table.
        grid_1_inner                       =   np.concatenate((sudoku_table_inner[0, 0], sudoku_table_inner[1, 0], sudoku_table_inner[1, 1],
                                                               sudoku_table_inner[2, 0], sudoku_table_inner[2, 1], sudoku_table_inner[3, 0]))
        grid_2_inner                       =   np.concatenate((sudoku_table_inner[3, 1], sudoku_table_inner[3, 2], sudoku_table_inner[4, 0],
                                                               sudoku_table_inner[4, 1], sudoku_table_inner[5, 0], sudoku_table_inner[5, 1]))
        grid_3_inner                       =   np.concatenate((sudoku_table_inner[0, 1], sudoku_table_inner[0, 2], sudoku_table_inner[0, 3],
                                                               sudoku_table_inner[1, 2], sudoku_table_inner[1, 3], sudoku_table_inner[2, 2]))
        grid_4_inner                       =   np.concatenate((sudoku_table_inner[3, 3], sudoku_table_inner[4, 2], sudoku_table_inner[4, 3],
                                                               sudoku_table_inner[5, 2], sudoku_table_inner[5, 3], sudoku_table_inner[5, 4]))
        grid_5_inner                       =   np.concatenate((sudoku_table_inner[0, 4], sudoku_table_inner[0, 5], sudoku_table_inner[1, 4],
                                                               sudoku_table_inner[1, 5], sudoku_table_inner[2, 3], sudoku_table_inner[2, 4]))
        grid_6_inner                       =   np.concatenate((sudoku_table_inner[2, 5], sudoku_table_inner[3, 4], sudoku_table_inner[3, 5],
                                                               sudoku_table_inner[4, 4], sudoku_table_inner[4, 5], sudoku_table_inner[5, 5]))
        self.sudoku_table_inner_batch      =   np.concatenate((sudoku_table_inner[:, :].reshape((6, 36)),
                                                   np.swapaxes(sudoku_table_inner[:, :], 0, 1).reshape((6, 36)),
                                                     np.array([grid_1_inner]),
                                                     np.array([grid_2_inner]),
                                                     np.array([grid_3_inner]),
                                                     np.array([grid_4_inner]),
                                                     np.array([grid_5_inner]),
                                                     np.array([grid_6_inner])))

        # generate rows, columns and grids of the resistors table.
        grid_1_resistor                         =   np.concatenate((sudoku_table_resistor[0, 0], sudoku_table_resistor[1, 0], sudoku_table_resistor[1, 1],
                                                                    sudoku_table_resistor[2, 0], sudoku_table_resistor[2, 1], sudoku_table_resistor[3, 0]))
        grid_2_resistor                         =   np.concatenate((sudoku_table_resistor[3, 1], sudoku_table_resistor[3, 2], sudoku_table_resistor[4, 0],
                                                                    sudoku_table_resistor[4, 1], sudoku_table_resistor[5, 0], sudoku_table_resistor[5, 1]))
        grid_3_resistor                         =   np.concatenate((sudoku_table_resistor[0, 1], sudoku_table_resistor[0, 2], sudoku_table_resistor[0, 3],
                                                                    sudoku_table_resistor[1, 2], sudoku_table_resistor[1, 3], sudoku_table_resistor[2, 2]))
        grid_4_resistor                         =   np.concatenate((sudoku_table_resistor[3, 3], sudoku_table_resistor[4, 2], sudoku_table_resistor[4, 3],
                                                                    sudoku_table_resistor[5, 2], sudoku_table_resistor[5, 3], sudoku_table_resistor[5, 4]))
        grid_5_resistor                         =   np.concatenate((sudoku_table_resistor[0, 4], sudoku_table_resistor[0, 5], sudoku_table_resistor[1, 4],
                                                                    sudoku_table_resistor[1, 5], sudoku_table_resistor[2, 3], sudoku_table_resistor[2, 4]))
        grid_6_resistor                         =   np.concatenate((sudoku_table_resistor[2, 5], sudoku_table_resistor[3, 4], sudoku_table_resistor[3, 5],
                                                                    sudoku_table_resistor[4, 4], sudoku_table_resistor[4, 5], sudoku_table_resistor[5, 5]))
        self.sudoku_table_resistor_batch        =   np.concatenate((sudoku_table_resistor[:, :].reshape((6, 36)),
                                                        np.swapaxes(sudoku_table_resistor[:, :], 0, 1).reshape((6, 36)),
                                                          np.array([grid_1_resistor]),
                                                          np.array([grid_2_resistor]),
                                                          np.array([grid_3_resistor]),
                                                          np.array([grid_4_resistor]),
                                                          np.array([grid_5_resistor]),
                                                          np.array([grid_6_resistor])))

        # deduce and generate inner values update for missing numbers in each row, column and grid by forward-feeding and back-propagation.
        layer_list = self.generate_values_for_each_layer(self.activator( self.sudoku_table_inner_batch))
        self.train_for_input_inner(layer_list, desired_output)

        # apply update to inner values for missing numbers in each row, column and grid.
        sudoku_table_inner[:, :] += self.sudoku_table_inner_batch_update[0:6].reshape((6, 6, 6))
        sudoku_table_inner[:, :] += np.swapaxes(self.sudoku_table_inner_batch_update[6:12].reshape((6, 6, 6)), 0, 1)
        sudoku_table_inner[0, 0] += self.sudoku_table_inner_batch_update[12][0 :6 ]
        sudoku_table_inner[1, 0] += self.sudoku_table_inner_batch_update[12][6 :12]
        sudoku_table_inner[1, 1] += self.sudoku_table_inner_batch_update[12][12:18]
        sudoku_table_inner[2, 0] += self.sudoku_table_inner_batch_update[12][18:24]
        sudoku_table_inner[2, 1] += self.sudoku_table_inner_batch_update[12][24:30]
        sudoku_table_inner[3, 0] += self.sudoku_table_inner_batch_update[12][30:36]

        sudoku_table_inner[3, 1] += self.sudoku_table_inner_batch_update[13][0 :6 ]
        sudoku_table_inner[3, 2] += self.sudoku_table_inner_batch_update[13][6 :12]
        sudoku_table_inner[4, 0] += self.sudoku_table_inner_batch_update[13][12:18]
        sudoku_table_inner[4, 1] += self.sudoku_table_inner_batch_update[13][18:24]
        sudoku_table_inner[5, 0] += self.sudoku_table_inner_batch_update[13][24:30]
        sudoku_table_inner[5, 1] += self.sudoku_table_inner_batch_update[13][30:36]

        sudoku_table_inner[0, 1] += self.sudoku_table_inner_batch_update[14][0 :6 ]
        sudoku_table_inner[0, 2] += self.sudoku_table_inner_batch_update[14][6 :12]
        sudoku_table_inner[0, 3] += self.sudoku_table_inner_batch_update[14][12:18]
        sudoku_table_inner[1, 2] += self.sudoku_table_inner_batch_update[14][18:24]
        sudoku_table_inner[1, 3] += self.sudoku_table_inner_batch_update[14][24:30]
        sudoku_table_inner[2, 2] += self.sudoku_table_inner_batch_update[14][30:36]

        sudoku_table_inner[3, 3] += self.sudoku_table_inner_batch_update[15][0 :6 ]
        sudoku_table_inner[4, 2] += self.sudoku_table_inner_batch_update[15][6 :12]
        sudoku_table_inner[4, 3] += self.sudoku_table_inner_batch_update[15][12:18]
        sudoku_table_inner[5, 2] += self.sudoku_table_inner_batch_update[15][18:24]
        sudoku_table_inner[5, 3] += self.sudoku_table_inner_batch_update[15][24:30]
        sudoku_table_inner[5, 4] += self.sudoku_table_inner_batch_update[15][30:36]

        sudoku_table_inner[0, 4] += self.sudoku_table_inner_batch_update[16][0 :6 ]
        sudoku_table_inner[0, 5] += self.sudoku_table_inner_batch_update[16][6 :12]
        sudoku_table_inner[1, 4] += self.sudoku_table_inner_batch_update[16][12:18]
        sudoku_table_inner[1, 5] += self.sudoku_table_inner_batch_update[16][18:24]
        sudoku_table_inner[2, 3] += self.sudoku_table_inner_batch_update[16][24:30]
        sudoku_table_inner[2, 4] += self.sudoku_table_inner_batch_update[16][30:36]

        sudoku_table_inner[2, 5] += self.sudoku_table_inner_batch_update[17][0 :6 ]
        sudoku_table_inner[3, 4] += self.sudoku_table_inner_batch_update[17][6 :12]
        sudoku_table_inner[3, 5] += self.sudoku_table_inner_batch_update[17][12:18]
        sudoku_table_inner[4, 4] += self.sudoku_table_inner_batch_update[17][18:24]
        sudoku_table_inner[4, 5] += self.sudoku_table_inner_batch_update[17][24:30]
        sudoku_table_inner[5, 5] += self.sudoku_table_inner_batch_update[17][30:36]


        return sudoku_table_inner


