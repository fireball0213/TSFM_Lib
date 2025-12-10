from selector.baseline_select import Baseline_Select_Model


class MyFancySelectModel(Baseline_Select_Model):

    def _get_select_strategy(self, dataset):
        test_dataset = dataset.test_data.input
        def select_strategy(dataset_name=None):
            # TODO: 实现一个新的选择逻辑
            raise NotImplementedError
        return select_strategy