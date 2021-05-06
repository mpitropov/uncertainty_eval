class Callback():
    def on_al_begin(self, loggers):
        pass

    def on_al_end(self):
        pass

    def on_cycle_begin(self, cycle):
        pass

    def on_cycle_end(self, cycle):
        pass

    def on_query_begin(self, cycle, pool_log_path, unlabeled_output_path):
        pass

    def on_query_end(self, cycle, queried_data, pool_log_path, unlabeled_output_path):
        pass

    def on_train_begin(self, cycle, model_path):
        pass

    def on_train_end(self, cycle, model_path):
        pass

    def on_eval_begin(self, cycle, test_output_path):
        pass

    def on_eval_end(self, cycle, eval_results, test_output_path):
        pass
