from dataloading import load_data
from utils import load_config


def do_training(epochs, model, data_obj, loss_func, opt, leave_out_index):
    model.train()
    for epoch in range(epochs):
        train_data = ...
        val_data = ...
        train_loss = train_step(model, train_data, loss_func, opt)
        val_loss = val_step(model, train_data, loss_func, opt)
        # print progress

    
    return val_loss


