import torch
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, NUM_WORKERS, VAL_BATCHSIZE, SHUFFLE_TRAIN, SHUFFLE_VAL
import utils
from torch.utils.data import Dataset, DataLoader


def retrieve_dataset(seed):
    """Dataset retrived is in the form (Torch tensor):
        [0]: image
        [1]: corresponding output value"""
    dataset = utils.create_raw_dataset(tensor=1, seed=seed)

    num_train = int(len(dataset) * TRAIN_RATIO)
    num_val = int(len(dataset) * VAL_RATIO)
    num_test = len(dataset) - num_train - num_val

    train_set = dataset[0:num_train]
    val_set = dataset[num_train:num_train + num_val]
    test_set = dataset[num_train + num_val: len(dataset)]

    return train_set, val_set, test_set, num_train, num_val, num_test


def dataset_import(inference=False, seed=0, model=None):
    train_dataset, valid_dataset, test_dataset, num_train, num_val, num_test = retrieve_dataset(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=VAL_BATCHSIZE,
        shuffle=SHUFFLE_VAL,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    if not inference:
        utils.write_run_configs(num_train, num_val, num_test, seed=seed, model=model)
        return train_loader, validation_loader, test_loader
    else:
        return test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = dataset_import()
    '''for X, y in train_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break'''
    #print(train_loader.dataset)
