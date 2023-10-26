"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.autograd import gradcheck

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import model.transformer as transformer
from evaluate import evaluate_lookup

from torchviz import make_dot

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/pidn',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--cuda', default='False',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")

def train_lookup(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

        Args:
            model: (torch.nn.Module) the neural network
            optimizer: (torch.optim) optimizer for parameters of model
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    torch.autograd.set_detect_anomaly(True)

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, all_batch) in enumerate(dataloader):
            # if params.cuda:
            #    train_batch, labels_batch, all_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True), all_batch.cuda(non_blocking=True)

            # convert to torch Variables
            # train_batch, labels_batch, all_batch = Variable(train_batch), Variable(labels_batch), Variable(all_batch)

            optimizer.zero_grad()

            #Vyt = all_batch[0][-1][6]
            #Yrt = all_batch[0][-1][3]

            #losses_vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
            #losses_yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)

            #vys = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
            #yrs = torch.zeros(labels_batch.size()[0], dtype=torch.float64)


            #model.init_hidden(train_batch.size(1))

            output_batch = model(train_batch)


            cf_loss, cr_loss = loss_fn(all_batch, output_batch)

            # loss = loss_vy_total
            # print(loss).

            loss = cf_loss + cr_loss

            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # utils.plot_grad_flow(model.named_parameters())

            # optimizer.zero_grad()
            '''
                # Manually update the parameters using gradients
                with torch.no_grad():
                    t -= optimizer.param_groups[0]['lr'] * t.grad

                    # Manually zero the gradients after updating the parameters
                    t.grad.zero_()

            loss_vy, loss_yr, _, _ = loss_fn(output, l, a, True, Vyt, Yrt)

            loss = 0.6*loss_vy + 0.4*loss_yr
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            '''

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                # print("output", output)
                # print("all", all[0, 4], all[0, 2])
                # print(losses_vy[-1])
                # print(Vyt, Yrt, label[1])
                # output_batch = output_batch.data.cpu().numpy()
                # labels_batch = labels_batch.data.cpu().numpy()
                # for p in model.parameters():
                #    print(p)
                #    break
                cf_loss = cf_loss.detach().numpy()
                cr_loss = cr_loss.detach().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](cf_loss, cr_loss) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.5f}'.format(loss_avg()))
            t.update()

    utils.plot_grad_flow_now()
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.8f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)



def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

        Args:
            model: (torch.nn.Module) the neural network
            optimizer: (torch.optim) optimizer for parameters of model
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    torch.autograd.set_detect_anomaly(True)

    hint_count = 1000

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, all_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch, all_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True), all_batch.cuda(non_blocking=True)

            # convert to torch Variables
            #train_batch, labels_batch, all_batch = Variable(train_batch), Variable(labels_batch), Variable(all_batch)

            optimizer.zero_grad()

            Vyt = all_batch[0][-1][7]
            Yrt = all_batch[0][-1][6]

            losses_vy = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
            losses_yr = torch.zeros(labels_batch.size()[0], dtype=torch.float64)

            vys = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
            yrs = torch.zeros(labels_batch.size()[0], dtype=torch.float64)
            #print(train_batch.size())

            #model.init_hidden(train_batch.size(1))

            output_batch = model(train_batch)


            for j, (output, label, all) in enumerate(zip(output_batch, labels_batch, all_batch)) :
                # move to GPU if available
                #if params.cuda:
                #    train, label, all = train.cuda(non_blocking=True), label.cuda(non_blocking=True), all.cuda(non_blocking=True)

                if j % hint_count == 0:
                    Vyt = all[0][7]
                    Yrt = all[0][6]
                # convert to torch Variables
                #train, label, all = Variable(train), Variable(label), Variable(all)
                #tr.requires_grad_(True)
                # compute model output and loss
                #output = model(tr)

                #print("output", output.grad)
                #temp_matrix = torch.ones(1, 2)
                #all[0, 6], all[0, 3] Vyt, Yrt
                loss_vy, loss_yr, Vyt, Yrt = loss_fn(output, label, all, Vyt, Yrt)
                #print("loss_vy", gradcheck(loss_fn, (output, label, all, Vyt, Yrt)))
                #print("loss_vy", loss_vy.grad)
                #a = torch.autograd.grad(output, train, temp_matrix, retain_graph=True)[0]
                #print(a.shape)
                #print(a)

                #loss_vy = (Vyt - label[1]) ** 2
                #loss_yr = (Yrt - label[0]) ** 2

                #vys[j] = Vyt
                #yrs[j] = Yrt
                #loss = 0.6 * loss_vy + 0.4 * loss_yr
                #loss.backward(retain_graph=True)
                #loss_vy.backward(retain_graph=True)
                #loss_yr.backward(retain_graph=True)

                losses_vy[j] = loss_vy
                losses_yr[j] = loss_yr
                Vyt = Vyt.detach()
                Yrt = Yrt.detach()
                '''
                Cf = output_batch[:, 0, 0]
                Cr = output_batch[:, 0, 1]
                Vx = batch_data[:, -1, 2]
                Vy = batch_data[:, -1, 6]
                Yawrate = batch_data[:, -1, 3]
                Sas = batch_data[:, -1, 5]
                '''

            #hint_count += 1
            #loss_vy = loss_fn(vys, labels_batch[:, 1])
            #loss_yr = loss_fn(yrs, labels_batch[:, 0])
            loss_vy_total = torch.mean(losses_vy, dtype=torch.float64)
            loss_yr_total = torch.mean(losses_yr, dtype=torch.float64)

            #loss_yr_total.backward(retain_graph=True)
            #loss_vy_total.backward(retain_graph=True)


            loss = 1.0 * loss_vy_total + 1.0 * loss_yr_total
            #loss = loss_vy_total
            #print(loss)


            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            #utils.plot_grad_flow(model.named_parameters())

            #optimizer.zero_grad()
            '''
                # Manually update the parameters using gradients
                with torch.no_grad():
                    t -= optimizer.param_groups[0]['lr'] * t.grad

                    # Manually zero the gradients after updating the parameters
                    t.grad.zero_()
            
            loss_vy, loss_yr, _, _ = loss_fn(output, l, a, True, Vyt, Yrt)

            loss = 0.6*loss_vy + 0.4*loss_yr
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            '''


            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                #print("output", output)
                #print("all", all[0, 4], all[0, 2])
                #print(losses_vy[-1])
                #print(Vyt, Yrt, label[1])
                #output_batch = output_batch.data.cpu().numpy()
                #labels_batch = labels_batch.data.cpu().numpy()
                loss_vy_total = loss_vy_total.detach().numpy()
                loss_yr_total = loss_yr_total.detach().numpy()
                #for p in model.parameters():
                #    print(p)
                #    break

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](loss_vy_total, loss_yr_total) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.5f}'.format(loss_avg()))
            t.update()

    utils.plot_grad_flow_now()
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.8f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 100000.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics, _, _, _, _, _, _, _, _, _ = evaluate(model, loss_fn, val_dataloader, metrics, params, True)

        val_acc = val_metrics['loss']
        is_best = val_acc <= best_val_acc

        print(val_acc ,best_val_acc)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path

        scheduler.step()
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    # loss function, train, evaluation
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')

    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available

    if args.cuda == 'False':
        params.cuda = False
    else :
        params.cuda = torch.cuda.is_available()
    #print("Is cuda available? " + str(torch.cuda.is_available()))
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Choose input features in data
    #selected_indices = [0, 1, 2, 6, 4]
    #selected_indices = [23, 24, 25, 29, 27]
    #selected_indices = [42, 43, 44, 46, 45]
    #selected_indices = [42, 43, 44, 46, 27]
    #selected_indices = [42, 44]  # 1
    selected_indices = [47, 48, 49, 51, 50]
    #selected_indices = [42, 44, 46]
    # 1 shuffle 2 un-shuffle

    # 8-12, 7-11, 10-14
    # Set input number of columns
    window = 1
    shift = 1

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, params, selected_indices, window, shift)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model
    #model = net.FourLayerModel(params).cuda() if params.cuda else net.FourLayerModel(params)
    #model = net.LSTMRegression(params).cuda() if params.cuda else net.LSTMRegression(params)
    model = net.AttentionRegressionModel(params=params, hidden_dim=512, num_heads=5, num_layers=2).cuda() \
        if params.cuda else net.AttentionRegressionModel(params=params, hidden_dim=512, num_heads=5, num_layers=2)
    '''
    # Model parameters
    dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 5 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 92 # length of input given to decoder. Can have any integer value.
    enc_seq_len = 153 # length of input given to encoder. Can have any integer value.
    output_sequence_length = 58 # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
    model = transformer.TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length, 
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)
    '''

    # Use parameters as float 64
    model.double()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.regularizer)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    #, weight_decay=params.regularizer

    # fetch loss function and metrics
    #loss_fn = net.loss_lookup
    loss_fn = net.loss_beta
    #loss_fn = net.loss_vy_yr
    metrics = net.metrics

    print(model)
    print("number of parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, scheduler, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
