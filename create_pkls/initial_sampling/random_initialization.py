import numpy as np

from initial_sampling import InitialSampling


class RandomInitializer(InitialSampling):

    def __init__(self, seed_num, pool, data_source, num_data):
        super().__init__(f'random_init_{num_data}', seed_num, pool, data_source, num_data )

    def data(self):
        """

        @return: return indices of unlabeled data selected uniformly randomly from the pool.
        The selected data may or may not be class balanced. It depends on the distribution of classes in the pool.
        """
        np.random.seed(self.seed_num)
        return np.random.choice(list(range(0, self.pool.unlabeled_data_size())), size=self.num_data, replace=False)