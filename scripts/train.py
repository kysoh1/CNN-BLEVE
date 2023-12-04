import copy

import torch
from torch import nn
from dataset import dataset_import
from utils import save_model, save_running_logs, load_model, create_run_dirs
from metrics import mean_absolute_percentage_error, RMSELoss
import sys
import numpy as np
from config import *
from model import create_model
import wandb

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if NP_FULL_SIZE:
    np.set_printoptions(threshold=sys.maxsize)


def model_param_tweaking(model):
    if MSE_REDUCTION == "mean" or MSE_REDUCTION == "sum":
        loss_func = nn.MSELoss(reduction=MSE_REDUCTION)
    else:
        loss_func = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if SCHEDULED:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                               factor=0.1, patience=PATIENCE, threshold=0.0001,
                                                               threshold_mode='abs')
    if SCHEDULED and VARRY_LR:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                               factor=0.1, patience=PATIENCE, threshold=0.0001,
                                                               threshold_mode='abs', min_lr=MIN_LEARNING_RATE)
    return loss_func, optimiser, scheduler


def get_best_val_model(curr_mape, best_mape, model, seed):
    if curr_mape < best_mape:
        write_info = "Best validation mape is {}, improved from {}".format(curr_mape, best_mape)
        save_running_logs(write_info, seed)
        return curr_mape
    else:
        return best_mape


def run_model(model, loss_func, dataloader, mode, print_tensor_output=False, best_mape=1.01, seed=0):
    size = len(dataloader)
    i = 0
    mape_sum = 0
    rmse_sum = 0
    loss_sum = 0
    acc_sum = 0

    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = get_predictions(X, model)

        loss_value = loss_func(pred, y)

        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        mape = mean_absolute_percentage_error(y, pred)

        rmse = RMSELoss(y, pred)
        acc = (1.0 - mape) * 100.0
        loss_value, current = loss_value.item(), batch * len(X)
        if print_tensor_output:
            print("pred: {}\ntru: {}".format(pred, y))

        write_info = f"{mode}:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}    accuracy: {acc:>4f}%"
        save_running_logs(write_info, seed)

        loss_sum += loss_value
        mape_sum += mape
        rmse_sum += rmse
        acc_sum += acc
        i += 1

    write_info = f"{mode} Summary:  avg_loss: {loss_sum / i:>7f}   avg_rmse: {rmse_sum / i:>0.4f}    avg_mape: {mape_sum / i:>0.4f}  avg_accuracy: {acc_sum / i:>4f}%"
    wandb_running_log(loss=loss_sum/i, mape=mape_sum/i, rmse=rmse_sum/i, accuracy=acc_sum/i, state=mode)
    save_running_logs(write_info, seed)
    if mode == "Validation":
        return mape_sum / i, loss_sum


def get_learning_rate(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']


def wandb_running_log(loss, accuracy, mape, rmse, state="Train"):
    wandb.log({f'{state}/loss': loss, f'{state}/rmse': rmse, f'{state}/mape': mape, f'{state}/accuracy': accuracy})


def empty_cuda_cache():
    if EMPTY_CUDA_CACHE:
        torch.cuda.empty_cache()


def test(test_dataloader, model, loss_func, seed):
    print("Final testing")
    # Start model testing
    if torch.cuda.is_available():
        empty_cuda_cache()
        with torch.no_grad():
            run_model(model, loss_func, test_dataloader, "Test", PRINT_TEST, seed=seed)

        empty_cuda_cache()
    else:
        run_model(model, loss_func, test_dataloader, "Test", PRINT_TEST, seed=seed)


def validation(val_dataloader, model, loss_func, best_mape, seed):
    # Start model evaluation
    if torch.cuda.is_available():
        empty_cuda_cache()
        with torch.no_grad():
            val_mape, val_loss = run_model(model, loss_func, val_dataloader, "Validation", PRINT_VAL, best_mape, seed=seed)

        empty_cuda_cache()
    else:
        val_mape, val_loss = run_model(model, loss_func, val_dataloader, "Validation", PRINT_VAL, best_mape, seed=seed)

    best_mape = get_best_val_model(curr_mape=val_mape, best_mape=best_mape, model=model, seed=seed)
    return best_mape, val_loss


def train(train_dataloader, model, loss_func, optimiser, seed):
    size = len(train_dataloader.dataset)
    total_loss = 0

    # Training
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass
        pred = get_predictions(X, model)

        loss_value = loss_func(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        # Backpropagation
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()
        total_loss += loss_value

        if batch % 100 == 0:
            mape = mean_absolute_percentage_error(y, pred)
            rmse = RMSELoss(y, pred)
            loss_value, current = loss_value.item(), batch * len(X)
            if PRINT_TRAIN:
                print("pred: {}\ntru: {}".format(pred, y))

            write_info = f"Train:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}"
            wandb_running_log(loss=loss_value, mape=mape, rmse=rmse, accuracy=(1.0 - mape)*100.0, state="Train")
            save_running_logs(write_info, seed)

    return total_loss


def get_predictions(input_data, model):
    if MODEL_NAME == "inceptionv3":
        predictions = model(input_data)[0].squeeze()
    else:
        predictions = model(input_data).squeeze()

    return predictions


def clone_model(og, clone):
    mp = list(og.parameters())
    mcp = list(clone.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]
    return clone


def restore_params(ckpt, model, optimiser):
    optimiser = optimiser.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    loss = ckpt['loss']
    return optimiser, epoch, loss


def main(seed):
    model = create_model().to(DEVICE)
    train_dataloader, validation_dataloader, test_loader = dataset_import(seed=seed, model=model)
    RESUME = False

    if seed == 0:
        print(model)

    loss_func, optimiser, lr_scheduler = model_param_tweaking(model)

    start_epoch = 0
    if RESUME:
        model, ckpt = load_model(LOAD_MODEL_LOCATION)
        optimiser, start_epoch, loss = restore_params(ckpt, model, optimiser)

    loss_func, optimiser, lr_scheduler = model_param_tweaking(model)
    best_model = copy.deepcopy(model)
    best_mape = 1.01

    epochs = NUM_EPOCHS
    for i in range(start_epoch, epochs):
        write_info = "___Epoch {}______________________________________________________________________".format(i + 1)
        save_running_logs(write_info, seed)

        train_loss = train(train_dataloader, model, loss_func, optimiser, seed)
        curr_lr = get_learning_rate(optimiser)
        wandb.log({'Train/lr': curr_lr})
        #wandb.watch(model)
        '''if SCHEDULED:
            lr_scheduler.step(train_loss)'''

        write_info = "________________________________________________________________________________\n"
        save_running_logs(write_info, seed)

        if i % SAVED_EPOCH == 0 and i > SAVED_AFTER:
            save_model(model, seed=seed, save_from_val=False, final=False, epoch=i, loss=train_loss, optimiser=optimiser)

        if i % 5 == 0 and i > 1:
            write_info = "---------------------------------VALIDATION AT EPOCH {}-----------------------------------".format(
                i + 1)
            save_running_logs(write_info, seed)

            returned_mape, val_loss = validation(validation_dataloader, model, loss_func, best_mape, seed)
            if best_mape > returned_mape:      # found a better mape
                save_model(model, seed=seed, save_from_val=True, final=False, epoch=i, loss=train_loss, optimiser=optimiser)
                best_model = copy.deepcopy(model)
                #best_model = clone_model(model, best_model)
                best_mape = returned_mape

            write_info = "------------------------------------END OF VALIDATION----------------------------------------\n"
            save_running_logs(write_info, seed)

            if SCHEDULED:
                lr_scheduler.step(best_mape)

    print("Training complete")
    save_model(model, seed=seed, final=True, optimiser=optimiser)
    test(test_loader, model, loss_func, seed)

    # Now run with best model
    loss_func, optimiser, lr_scheduler = model_param_tweaking(best_model)
    save_running_logs("Running with best val model:", seed)
    test(test_loader, best_model, loss_func, seed)


if __name__ == '__main__':
    create_run_dirs()
    for seed in range (0, SEED_RANGE):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wandb_run_name = f"{WANDB_PROJECT_NAME}_seed_{seed}"
        wandb.init(project=wandb_run_name)
        main(seed)
        wandb.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    #perform_inference()
