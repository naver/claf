
from overrides import overrides
from torch.utils.data import DataLoader

from .base import Factory


class DataLoaderFactory(Factory):
    """
    DataLoader Factory Class

    * Args:
        config: data_loader config from argument (config.data_loader)
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.cuda_device_id = None
        if config.cuda_devices:
            self.cuda_device_id = config.cuda_devices[0]

    @overrides
    def create(self, datasets):
        """ create train, valid and test iterator """
        dataset_key = next(iter(datasets))
        dataset = datasets[dataset_key]

        if getattr(dataset, "name", None) is None:
            raise ValueError("unknown dataset.")

        train_loader = None
        if "train" in datasets:
            train_loader = self.make_data_loader(
                datasets["train"],
                batch_size=self.batch_size,
                shuffle=True,
                cuda_device_id=self.cuda_device_id,
            )
        valid_loader = None
        if "valid" in datasets:
            valid_loader = self.make_data_loader(
                datasets["valid"],
                batch_size=self.batch_size // 2,
                shuffle=False,
                cuda_device_id=self.cuda_device_id,
            )
        test_loader = None
        if "test" in datasets:
            test_loader = self.make_data_loader(
                datasets["test"],
                batch_size=self.batch_size,
                shuffle=False,
                cuda_device_id=self.cuda_device_id,
            )
        return train_loader, valid_loader, test_loader

    def make_data_loader(self, dataset, batch_size=32, shuffle=True, cuda_device_id=None):
        is_cpu = cuda_device_id is None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn(cuda_device_id=cuda_device_id),
            num_workers=0,
            pin_memory=is_cpu,  # only CPU memory can be pinned
        )
