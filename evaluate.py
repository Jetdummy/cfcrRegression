"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/pidn',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params, train):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    Vy_true = np.empty((0))
    Yr_true = np.empty((0))
    y_pred = np.empty((1, 2))
    Vy_pred = np.empty((0))
    Yr_pred = np.empty((0))
    Cf_lookup = np.empty((0))
    Cr_lookup = np.empty((0))
    Vy_lookup = np.empty((0))
    Yr_lookup = np.empty((0))


    # compute metrics over the dataset
    for i, (data_batch, labels_batch, all_batch) in enumerate(dataloader):
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch, all_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True), all_batch.cuda(non_blocking=True)
        if i % 512 == 0 and not train:
            Vyt = all_batch[0][-1][7]#6
            Yrt = all_batch[0][-1][6]
            Vy = all_batch[0][-1][7].item()
            Yr = all_batch[0][-1][6].item()
            #model.init_hidden(data_batch.size(1))
        if train:
            Vyt = all_batch[0][-1][7]
            Yrt = all_batch[0][-1][6]



        losses_vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
        losses_yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)

        predict_Vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
        predict_Yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
        if params.cuda:
            losses_vy, losses_yr, predict_Vy, predict_Yr = losses_vy.cuda(non_blocking=True), losses_yr.cuda(
                non_blocking=True), predict_Vy.cuda(non_blocking=True), predict_Yr.cuda(non_blocking=True)

        output_batch = model(data_batch)

        for j, (output, label, all) in enumerate(zip(output_batch, labels_batch, all_batch)):
            if params.cuda:
                output, label, all = output.cuda(non_blocking=True), label.cuda(non_blocking=True), all.cuda(
                    non_blocking=True)

            loss_vy, loss_yr, Vyt, Yrt = loss_fn(output, label, all, Vyt, Yrt)

            #losses_vy[j] = (Vyt - label[1]) ** 2
            #losses_yr[j] = (Yrt - label[0]) ** 2
            losses_vy[j] = loss_vy
            losses_yr[j] = loss_yr

            Vyt = Vyt.detach()
            Yrt = Yrt.detach()

            predict_Vy[j] = Vyt
            predict_Yr[j] = Yrt

        loss_vy = torch.mean(losses_vy, dtype=torch.float64)
        loss_yr = torch.mean(losses_yr, dtype=torch.float64)

        loss = 1.0 * loss_vy + 1.0 * loss_yr
        #loss = loss_vy
        '''
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch, all_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True), all_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch, all_batch = Variable(data_batch), Variable(labels_batch), Variable(all_batch)

        if first and not train :
            predict_Vy = all_batch[0][-1][6]
            predict_Yr = all_batch[0][-1][3]
            first = False
        if train :
            predict_Vy = all_batch[0][-1][6]
            predict_Yr = all_batch[0][-1][3]
        # compute model output
        output_batch = model(data_batch)

        loss_vy, loss_yr, predict_Vy, predict_Yr = loss_fn(output_batch, labels_batch, all_batch, train, predict_Vy, predict_Yr)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        loss = loss_vy * 0.6 + loss_yr * 0.4

        loss_vy = loss_vy.detach()
        loss_yr = loss_yr.detach()
        '''
        output_batch = output_batch.detach()
        loss = loss.cpu().detach()
        loss_vy = loss_vy.cpu().detach()
        loss_yr = loss_yr.cpu().detach()


        if output_batch.ndim == 1:
            output_batch = output_batch.reshape(1, 2)
        else:
            squeezed_array = np.squeeze(output_batch)
            output_batch = squeezed_array.reshape((-1, 2))


        if not train:
            with torch.no_grad():
                # Concat output and label
                Vy_true = np.concatenate((Vy_true, labels_batch[:, 1]), axis=0)
                Yr_true = np.concatenate((Yr_true, labels_batch[:, 0]), axis=0)
                y_pred = np.concatenate((y_pred, output_batch), axis=0)
                Vy_pred = np.append(Vy_pred, predict_Vy.cpu().detach().numpy())
                Yr_pred = np.append(Yr_pred, predict_Yr.cpu().detach().numpy())
                cf = utils.lookup_table_cf(all_batch[0, 0, 1])
                cr = utils.lookup_table_cr(all_batch[0, 0, 1])
                Vy = utils.bicycle_beta(cf, cr, all_batch[0, 0, 4], Vy, Yr, all_batch[0, 0, 2])
                Yr = utils.bicycle_yr2(cf, cr, all_batch[0, 0, 4], Vy, Yr, all_batch[0, 0, 2])
                Cf_lookup = np.append(Cf_lookup, cf)
                Cr_lookup = np.append(Cr_lookup, cr)
                Vy_lookup = np.append(Vy_lookup, Vy)
                Yr_lookup = np.append(Yr_lookup, Yr)



        # compute all metrics on this batch
        with torch.no_grad():
            summary_batch = {metric: metrics[metric](loss_vy, loss_yr)
                            for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, Vy_true, Vy_pred, Yr_true, Yr_pred, y_pred, Cf_lookup, Cr_lookup, Vy_lookup, Yr_lookup


def evaluate_lookup(model, loss_fn, dataloader, metrics, params, train):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    Vy_true = np.empty((0))
    Yr_true = np.empty((0))
    y_pred = np.empty((1, 2))
    Vy_pred = np.empty((0))
    Yr_pred = np.empty((0))
    Cf_lookup = np.empty((0))
    Cr_lookup = np.empty((0))
    Vy_lookup = np.empty((0))
    Yr_lookup = np.empty((0))

    first = True

    # compute metrics over the dataset
    for data_batch, labels_batch, all_batch in dataloader:

        if first and not train:
            Vyt = all_batch[0][-1][6]
            Yrt = all_batch[0][-1][3]
            Vy = all_batch[0][-1][6].item()
            Yr = all_batch[0][-1][3].item()
            #model.init_hidden(data_batch.size(1))
            first = False
        if train:
            Vyt = all_batch[0][-1][6]
            Yrt = all_batch[0][-1][3]



        #losses_vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
        #losses_yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)

        #ddd
        #predict_Vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
        #predict_Yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)

        output_batch = model(data_batch)

        cf_loss, cr_loss = loss_fn(all_batch, output_batch)

        #ddd
        '''
        for j, (output, label, all) in enumerate(zip(output_batch, labels_batch, all_batch)):
            if params.cuda:
                output, label, all = output.cuda(non_blocking=True), label.cuda(non_blocking=True), all.cuda(
                    non_blocking=True)

            _, _, Vyt, Yrt = net.loss_vy_yr(output, label, all, Vyt, Yrt)

            #losses_vy[j] = (Vyt - label[1]) ** 2
            #losses_yr[j] = (Yrt - label[0]) ** 2
            #losses_vy[j] = loss_vy
            #losses_yr[j] = loss_yr

            Vyt = Vyt.detach()
            Yrt = Yrt.detach()

            predict_Vy[j] = Vyt
            predict_Yr[j] = Yrt
        '''
        #loss_vy = torch.mean(losses_vy, dtype=torch.float64)
        #loss_yr = torch.mean(losses_yr, dtype=torch.float64)

        loss = cf_loss + 0.4 * cr_loss
        #loss = loss_vy
        '''
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch, all_batch = data_batch.cuda(
                non_blocking=True), labels_batch.cuda(non_blocking=True), all_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch, all_batch = Variable(data_batch), Variable(labels_batch), Variable(all_batch)

        if first and not train :
            predict_Vy = all_batch[0][-1][6]
            predict_Yr = all_batch[0][-1][3]
            first = False
        if train :
            predict_Vy = all_batch[0][-1][6]
            predict_Yr = all_batch[0][-1][3]
        # compute model output
        output_batch = model(data_batch)

        loss_vy, loss_yr, predict_Vy, predict_Yr = loss_fn(output_batch, labels_batch, all_batch, train, predict_Vy, predict_Yr)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        loss = loss_vy * 0.6 + loss_yr * 0.4

        loss_vy = loss_vy.detach()
        loss_yr = loss_yr.detach()
        '''
        output_batch = output_batch.detach()

        cf_loss = cf_loss.detach()
        cr_loss = cr_loss.detach()


        if output_batch.ndim == 1:
            output_batch = output_batch.reshape(1, 2)
        else:
            squeezed_array = np.squeeze(output_batch)
            output_batch = squeezed_array.reshape((-1, 2))


        if not train:
            with torch.no_grad():
                # Concat output and label
                Vy_true = np.concatenate((Vy_true, labels_batch[:, 1]), axis=0)
                Yr_true = np.concatenate((Yr_true, labels_batch[:, 0]), axis=0)
                y_pred = np.concatenate((y_pred, output_batch), axis=0)
                #ddd
                #Vy_pred = np.append(Vy_pred, predict_Vy.cpu().detach().numpy())
                #Yr_pred = np.append(Yr_pred, predict_Yr.cpu().detach().numpy())
                cf = utils.lookup_table_cf(all_batch[0, 0, 1])
                cr = utils.lookup_table_cr(all_batch[0, 0, 1])
                Vy = utils.bicycle_Vy2(cf, cr, all_batch[0, 0, 4], Vy, Yr, all_batch[0, 0, 2])
                Yr = utils.bicycle_yr2(cf, cr, all_batch[0, 0, 4], Vy, Yr, all_batch[0, 0, 2])
                Cf_lookup = np.append(Cf_lookup, cf)
                Cr_lookup = np.append(Cr_lookup, cr)
                Vy_lookup = np.append(Vy_lookup, Vy)
                Yr_lookup = np.append(Yr_lookup, Yr)



        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](cf_loss, cr_loss)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, Vy_true, Vy_pred, Yr_true, Yr_pred, y_pred, Cf_lookup, Cr_lookup, Vy_lookup, Yr_lookup


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = False #torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Choose input features in data
    selected_indices = [23, 24, 25, 29, 27]
    #selected_indices = [42, 43, 44, 46, 45]
    # Set input number of columns
    window = 1
    shift = 1

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params, selected_indices, window, shift)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    #model = net.FourLayerModel(params).cuda() if params.cuda else net.FourLayerModel(params)
    #model = net.LSTMRegression(params).cuda() if params.cuda else net.LSTMRegression(params)
    model = net.AttentionRegressionModel(params=params, hidden_dim=512, num_heads=1, num_layers=2).cuda() \
        if params.cuda else net.AttentionRegressionModel(params=params, hidden_dim=512, num_heads=1, num_layers=2)

    # Use parameters as float 64
    model.double()

    loss_fn = net.loss_beta
    #loss_fn = net.loss_vy_yr
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics, Vy_true, Vy_pred, Yr_true, Yr_pred, y_pred, Cf_look, Cr_look, Vy_look, Yr_look = evaluate(model, loss_fn, test_dl, metrics, params, False)
    utils.compare_as_plot_vy(Vy_true, Vy_pred, Vy_look)
    utils.compare_as_plot_yr(Yr_true, Yr_pred, Yr_look)
    utils.CfCr_as_plot(y_pred, Cf_look, Cr_look)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)