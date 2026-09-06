import inspect 
import importlib 
import pytorch_lightning as pl

from core.datamodule_policies import get_datamodule_policy


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.dataset_name = dataset_name
        self.data_config = kwargs['data_config']
        self.policy = get_datamodule_policy(self.data_config.get("datamodule_policy", None), dataset_name)
        self.load_data_module()

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        
        if stage == 'fit':
            self.train_dataset = self.instancialize(phase='train')
            self.val_dataset = self.instancialize(phase='test')

        if stage == 'test':
            self.test_dataset = self.instancialize(phase='test')
            
        if stage == "predict":
            self.test_dataset = self.instancialize(phase='test')


    def train_dataloader(self):
        return self.policy.train_dataloader(self.train_dataset, self.data_config)

    def val_dataloader(self):
        return self.policy.val_dataloader(self.val_dataset, self.data_config)

    def test_dataloader(self):
        return self.policy.test_dataloader(self.test_dataset, self.data_config)

    def predict_dataloader(self):
        return self.policy.predict_dataloader(self.test_dataset, self.data_config)

    def load_data_module(self):
        if '_' in self.dataset_name:
            camel_name = ''.join([i.capitalize() for i in self.dataset_name.split('_')])
        else:
            camel_name = self.dataset_name
        
        try:
            # self.data_module = getattr(importlib.import_module(
            #     f'datasets.{self.dataset_name}'), camel_name)
            self.data_module = getattr(importlib.import_module(
                f'dataset'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.data_config dictionary. You can also input any args
            to overwrite the corresponding value in self.data_config.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.data_config.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.data_config[arg]
        args1.update(other_args)
        return self.data_module(**args1)
