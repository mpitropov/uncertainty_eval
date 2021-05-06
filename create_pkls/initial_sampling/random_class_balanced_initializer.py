import numpy as np

from initial_sampling import InitialSampling


class RandomClassBalancedInitializer(InitialSampling):

    def __init__(self, seed_num, pool, data_source, num_data):
        super().__init__("random_initializer", seed_num, pool, data_source, num_data)

    def data(self):
        """
        @return: return indices of unlabeled data selected uniformly randomly from the pool.
        The selected data are class class balanced.
        """
        # TODO: This method should be implemented. The implementation should work for different tasks such as
        #  classification, object detection, semantic segmentation
        pass
