from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PygDataLoader


class DataModulePolicy:
    def train_dataloader(self, dataset, data_config):
        return self._torch_loader(dataset, data_config.train_dataloader)

    def val_dataloader(self, dataset, data_config):
        return self._torch_loader(dataset, data_config.test_dataloader)

    def test_dataloader(self, dataset, data_config):
        return self._torch_loader(dataset, data_config.test_dataloader)

    def predict_dataloader(self, dataset, data_config):
        return self._torch_loader(dataset, data_config.test_dataloader)

    @staticmethod
    def _torch_loader(dataset, loader_config):
        return DataLoader(
            dataset,
            batch_size=loader_config.batch_size,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
            shuffle=loader_config.shuffle,
        )


class GraphDataModulePolicy(DataModulePolicy):
    def train_dataloader(self, dataset, data_config):
        return self._pyg_loader(dataset, data_config.train_dataloader)

    def val_dataloader(self, dataset, data_config):
        return self._pyg_loader(dataset, data_config.test_dataloader)

    def test_dataloader(self, dataset, data_config):
        return self._pyg_loader(dataset, data_config.test_dataloader)

    def predict_dataloader(self, dataset, data_config):
        return self._pyg_loader(dataset, data_config.test_dataloader)

    @staticmethod
    def _pyg_loader(dataset, loader_config):
        return PygDataLoader(
            dataset,
            batch_size=loader_config.batch_size,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
            shuffle=loader_config.shuffle,
        )


class DirectDatasetPolicy(GraphDataModulePolicy):
    def val_dataloader(self, dataset, data_config):
        return dataset

    def test_dataloader(self, dataset, data_config):
        return dataset

    def predict_dataloader(self, dataset, data_config):
        return dataset


_POLICIES = {
    "default": DataModulePolicy,
    "graph": GraphDataModulePolicy,
    "direct": DirectDatasetPolicy,
}
_DATASET_FALLBACKS = {}


def register_datamodule_policy(name, policy_cls):
    _POLICIES[name] = policy_cls


def register_dataset_policy_fallback(dataset_name, policy_cls):
    _DATASET_FALLBACKS[dataset_name] = policy_cls


def get_datamodule_policy(policy_name=None, dataset_name=None):
    if policy_name:
        if policy_name not in _POLICIES:
            raise ValueError(f"Unknown datamodule policy: {policy_name}")
        return _POLICIES[policy_name]()
    return _DATASET_FALLBACKS.get(dataset_name, DataModulePolicy)()
