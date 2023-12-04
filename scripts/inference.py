import torch
from dataset import dataset_import
from utils import load_model
from config import *
from metrics import *


def get_predictions(input_data, model):
    if MODEL_NAME == "inceptionv3":
        predictions = model(input_data)[0].squeeze()
    else:
        predictions = model(input_data).squeeze()

    return predictions


def perform_inference(model, inf_on_test_set=False):
    if inf_on_test_set:
        dataloader = dataset_import(inference=True)  # optional

    # loss_func, optimiser, lr_scheduler = model_param_tweaking(model)
    model.eval()
    x, y = dataloader.dataset[0][0], dataloader.dataset[0][1]
    if DEVICE == 'cuda':
        with torch.no_grad():
            pred = get_predictions(x, model)
            mape = mean_absolute_percentage_error(y, pred)

            rmse = RMSELoss(y, pred)
            acc = (1.0 - mape) * 100.0
            print(f"Inference results: rmse: {rmse:>0.4f}    mape: {mape:>0.4f}    accuracy: {acc:>4f}%")
    else:
        pred = get_predictions(x, model)
        mape = mean_absolute_percentage_error(y, pred)

        rmse = RMSELoss(y, pred)
        acc = (1.0 - mape) * 100.0
        print(f"Inference results: rmse: {rmse:>0.4f}    mape: {mape:>0.4f}    accuracy: {acc:>4f}%")


if __name__ == '__main__':
    model = load_model(LOAD_MODEL_LOCATION, resume=False)
    perform_inference(model=model, inf_on_test_set=True)
