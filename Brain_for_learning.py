import numpy as np
from scipy.special import expit


class Brain(object):
    def __init__(self, network_size, slope, alpha, epoch_of_learning):

        self.network_size                 = network_size
        self.number_of_layers             = self.network_size.shape[0]

        self.slope                        = slope
        self.alpha                        = alpha
        self.epoch_of_learning            = epoch_of_learning

        self.weight_list                  = self.initialize_weight_list()
        self.slope_list                   = self.initialize_slope_list()


    def initialize_weight_list(self):
        weight_list = list()
        for i in range(self.number_of_layers - 1):
            weight                                              = (np.random.random((self.network_size[i]                                 , self.network_size[i+1]                          )) -0.5 ) * 0.1
            weight_list.append(weight)
        weight_list = np.asarray(weight_list)
        return  weight_list


    def initialize_slope_list(self):
        slope_list = list()
        for i in range(self.number_of_layers - 1):
            slope                                               = np.ones(self.network_size[i + 1]) * self.slope
            slope_list.append(slope)
        slope_list = np.asarray(slope_list)
        return slope_list


    def activator(self, x):
        return expit(x)


    def activator_output_to_derivative(self, output):
        return output * (1 - output)


    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        for i in range(self.number_of_layers - 1):

            layer                 = self.activator(np.dot(layer_list[-1]                          , self.weight_list[i]                                                          ) * self.slope_list[i] )

            layer_list.append(layer)

        return   layer_list


    def train_for_weight(self,
                       layer_list,
                       output):

        layer_delta_list       = list()
        slope_delta_list       = list()

        layer_final_error      = output - layer_list[-1]

        layer_delta            = (layer_final_error                                                                                               )    * self.activator_output_to_derivative(layer_list[-1] )          *   self.slope_list[-1]
        slope_delta            = (layer_final_error                                                                                               )    * self.activator_output_to_derivative(layer_list[-1] )          * np.dot(layer_list[-2], self.weight_list[-1])

        layer_delta_list.append(layer_delta)
        slope_delta_list.append(slope_delta)

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta_list[-1].dot(self.weight_list[- 1 - i].T                                                         ) )    * self.activator_output_to_derivative(layer_list[- 1 - 1 - i] )  *   self.slope_list[-1 -1 -i]
            slope_delta        = (layer_delta_list[-1].dot(self.weight_list[- 1 - i].T                                                         ) )    * self.activator_output_to_derivative(layer_list[- 1 - 1 - i] )  * np.dot(layer_list[-1 -1 -i -1], self.weight_list[-1 -i -1])

            layer_delta_list.append(layer_delta)
            slope_delta_list.append(slope_delta)

        for i in range(self.number_of_layers - 1):

            self.weight_list[i]                             += np.asarray(layer_list[i]                                                           ).T.dot(np.asarray(layer_delta_list[- 1 - i])                                        ) * self.alpha
            self.slope_list[i]                              += np.array(slope_delta_list[- 1 - i]).sum(axis=0)                                                                                                                           * self.alpha


    def learn_batch(self, input_list, output_list):

        layer_list = self.generate_values_for_each_layer(input_list)

        self.train_for_weight(
                   layer_list,
                   output_list)

        return self

