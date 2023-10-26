"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

import utils

class AttentionRegressionModel(nn.Module):
    def __init__(self, params, hidden_dim, num_heads, num_layers):
        super(AttentionRegressionModel, self).__init__()
        # Multi-Head Self-Attention을 사용하는 Transformer 인코더를 정의합니다.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.num_channels,
            nhead=num_heads,
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )
        # 최종 회귀 레이어를 정의합니다.
        self.regression_layer = nn.Linear(params.num_channels, 2)

    def forward(self, x):
        # 입력 데이터를 Transformer 인코더에 전달합니다.
        x = self.transformer_encoder(x)
        # 최종 회귀 레이어를 통해 예측을 수행합니다.
        output = self.regression_layer(x)
        output = torch.sigmoid(output)
        return output


class LSTMRegression(nn.Module):
    def __init__(self, params):
        super(LSTMRegression, self).__init__()
        self.num_channels = params.num_channels
        self.hidden_size = 20
        self.num_layers = 2

        self.lstm = nn.LSTM(self.num_channels, self.hidden_size, self.num_layers, bidirectional=False, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, 2)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64)
        #print(hidden_state.dtype)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        #h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        #print(x.dtype)
        out, self.hidden = self.lstm(x, self.hidden)
        #out = self.fc(out[:, -1, :])
        #out = torch.sigmoid(out)
        return out



class FourLayerModel(nn.Module):
    def __init__(self, params):
        self.num_channels = params.num_channels
        super(FourLayerModel, self).__init__()
        self.fc1 = nn.Linear(self.num_channels, 64)   # num_channels inputs -> 64 hidden units in the first layer
        #self.bn1 = nn.BatchNorm1d(num_features=64)
        self.do1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 128) # 64 hidden units -> 128 hidden units in the second layer
        #self.bn2 = nn.BatchNorm1d(num_features=128)
        self.do2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 128) # 128 hidden units -> 64 hidden units in the third layer
        #self.bn3 = nn.BatchNorm1d(num_features=128)
        self.do3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 64)  # 128 hidden units -> 64 hidden units in the third layer
        #self.bn4 = nn.BatchNorm1d(num_features=64)
        self.do4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(64, 2)   # 64 hidden units -> 2 output in the fourth layer
        self.double()
        self.activation = torch.nn.LeakyReLU()
        #torch.nn.init.xavier_uniform_(self.fc1.weight)
        #torch.nn.init.xavier_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.fc3.weight)
        #torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        '''
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        '''
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        #x = self.fc4(x)
        '''
        #torch.nn.ELU()
        #torch.nn.ReLU()
        #torch.nn.LeakyReLU()

        x = self.fc1(x)
        x = self.activation(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
        x = self.do1(x)
        x = self.fc2(x)
        x = self.activation(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
        x = self.do2(x)
        x = self.fc3(x)
        x = self.activation(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
        x = self.do3(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.do4(x)
        x = self.fc5(x)

        x = torch.sigmoid(x)
        '''
        if x.size()[0] == 1:
            x = self.fc1(x)
            #x = x.squeeze(1)
            #x = self.bn1(x)

            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do1(x)
            x = self.fc2(x)

            #x = self.bn2(x)

            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do2(x)
            #x = self.fc3(x)

            #x = self.bn3(x)

            #x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            #x = self.do3(x)
            x = self.fc4(x)

            #x = self.bn4(x)

            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do4(x)
            x = self.fc5(x)

            x = torch.sigmoid(x)
        else :
            x = self.fc1(x)
            #x = x.squeeze(1)  # Squeeze the second dimension to have shape [16, 64]
            #x = self.bn1(x)
            #x = x.unsqueeze(1)  # Permute dimensions back to [16, 1, 64]
            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do1(x)
            x = self.fc2(x)
            #x = x.squeeze(1)  # Squeeze the second dimension to have shape [16, 128]
            #x = self.bn2(x)
            #x = x.unsqueeze(1)  # Permute dimensions back to [16, 1, 128]
            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do2(x)
            #x = self.fc3(x)
            #x = x.squeeze(1)  # Squeeze the second dimension to have shape [16, 64]
            #x = self.bn3(x)
            #x = x.unsqueeze(1)  # Permute dimensions back to [16, 1, 64]
            #x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            #x = self.do3(x)
            x = self.fc4(x)
            #x = x.squeeze(1)  # Squeeze the second dimension to have shape [16, 64]
            #x = self.bn4(x)
            #x = x.unsqueeze(1)  # Permute dimensions back to [16, 1, 64]
            x = torch.relu(x)  # Alternatively, you can use nn.ReLU() instead of torch.relu
            x = self.do4(x)
            x = self.fc5(x)
            x = torch.sigmoid(x)
        '''

        return x


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8 * 8 * self.num_channels * 4, self.num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 6)
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 8 * 8 * self.num_channels * 4)  # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        s = self.fc2(s)  # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)


class bicycle_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        순전파 단계에서는 입력을 갖는 텐서를 받아 출력을 갖는 텐서를 반환합니다.
        ctx는 컨텍스트 객체(context object)로 역전파 연산을 위한 정보 저장에 사용합니다.
        ctx.save_for_backward 메소드를 사용하여 역전파 단계에서 사용할 어떤 객체도
        저장(cache)해 둘 수 있습니다.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파 단계에서는 출력에 대한 손실(loss)의 변화도(gradient)를 갖는 텐서를 받고,
        입력에 대한 손실의 변화도를 계산해야 합니다.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


def MSE_loss_fn(outputs, labels, batch_data, train, Vy, Yr):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
        batch_data : (Variable) dimension batch_size X all indices
        train: (Boolean) if train, previous Vy is ground truth

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    '''
    [batch_size][window_size][number of features]
    Cf = outputs[][0]
    Cr = outputs[][1]
    Vy_RT_loss = batch_data[][][6]
    yawrate_loss = batch_data[][][3]
    sas_loss = batch_data[][][2]
    Vy_RT_dot_loss = batch_data[][][6]
    v0(Vx) = batch_data[][][4]
    
    
    
    Vy_dot_est(1, i) = (-(Cf(1, i) + Cr(1, i)) / (M_nom * v0). * Vy_RT_loss(1, i) +
                        ((Lr_nom * Cr(1, i) - Lf_nom * Cf(1, i)) / (M_nom * v0) - v0). * yawrate_loss(1, i) +
                        Cf(1, i). * sas_loss(1, i) / M_nom)*T
    '''
    num_examples = batch_data.size()[0]  # batch size
    window_size = batch_data.size()[1]  # data number of columns

    loss_vy = torch.zeros(num_examples)
    loss_yr = torch.zeros(num_examples)

    # Vy를 출력해주기 위한 변수
    if train :
        for i in range(num_examples):
            Vy = utils.bicycle_Vy1(outputs[i, 0, 0], outputs[i, 0, 1], batch_data[i, 0, 4], Vy, Yr, batch_data[i, 0, 2])
            Yr = utils.bicycle_yr1(outputs[i, 0, 0], outputs[i, 0, 1], batch_data[i, 0, 4], Vy, Yr, batch_data[i, 0, 2])
            loss_vy[i] = (Vy - labels[i, 1]) ** 2
            loss_yr[i] = (Yr - labels[i, 0]) ** 2

    else :
        Vy = utils.bicycle_Vy1(outputs[:, :, 0], outputs[:, :, 1], batch_data[:, :, 4],
                                            Vy, Yr, batch_data[:, :, 2])
        Yr = utils.bicycle_yr1(outputs[:, :, 0], outputs[:, :, 1], batch_data[:, :, 4],
                                            Vy, Yr, batch_data[:, :, 2])

    #return (0.4 * loss_vy + 0.6 * loss_yr) / num_examples, Vy, Yr
    return loss_vy / num_examples, loss_yr / num_examples, Vy, Yr


def loss_vy_yr(output, label, batch_data, Vy, Yr):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
        batch_data : (Variable) dimension batch_size X all indices
        train: (Boolean) if train, previous Vy is ground truth

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    '''
    [batch_size][window_size][number of features]
    Cf = outputs[0]
    Cr = outputs[1]
    Vy_RT_loss = batch_data[][6]
    yawrate_loss = batch_data[][3]
    sas_loss = batch_data[][2]
    Vy_RT_dot_loss = batch_data[][6]
    v0(Vx) = batch_data[][4]



    Vy_dot_est(1, i) = (-(Cf(1, i) + Cr(1, i)) / (M_nom * v0). * Vy_RT_loss(1, i) +
                        ((Lr_nom * Cr(1, i) - Lf_nom * Cf(1, i)) / (M_nom * v0) - v0). * yawrate_loss(1, i) +
                        Cf(1, i). * sas_loss(1, i) / M_nom)*T
    '''
    window_size = batch_data.size()[1]  # data number of columns
    if output.dim() == 1:
        Vy_next = utils.bicycle_Vy2(output[0], output[1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])
        Yr_next = utils.bicycle_yr2(output[0], output[1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])
    else :
        Vy_next = utils.bicycle_Vy2(output[0, 0], output[0, 1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])
        Yr_next = utils.bicycle_yr2(output[0, 0], output[0, 1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])

    #print(Vy, label[1])
    loss_vy = torch.abs(Vy_next - label[1])
    loss_yr = torch.abs(Yr_next - label[0])
    '''
    print('vy error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Vy_next.item()) + str(' ') + \
                                       str(label[1].item()) + str(' ') + str(loss_vy.item()))
    
    assert abs(Vy_next.item()) < 100000, 'vy error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Vy_next.item()) + str(' ') + \
                                       str(label[1].item()) + str(' ') + str(loss_vy.item())
    assert abs(Yr_next.item()) < 100000, 'yr error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Yr_next.item()) + str(' ') + \
                                       str(label[0].item()) + str(' ') + str(loss_yr.item())
    '''
    # return (0.4 * loss_vy + 0.6 * loss_yr) / num_examples, Vy, Yr
    return loss_vy, loss_yr, Vy_next, Yr_next


def loss_beta(output, label, batch_data, Beta, Yr):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
        batch_data : (Variable) dimension batch_size X all indices
        train: (Boolean) if train, previous Vy is ground truth

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    '''
    [batch_size][window_size][number of features]
    Cf = outputs[0]
    Cr = outputs[1]
    Vy_RT_loss = batch_data[][6]
    yawrate_loss = batch_data[][3]
    sas_loss = batch_data[][2]
    Vy_RT_dot_loss = batch_data[][6]
    v0(Vx) = batch_data[][4]



    Vy_dot_est(1, i) = (-(Cf(1, i) + Cr(1, i)) / (M_nom * v0). * Vy_RT_loss(1, i) +
                        ((Lr_nom * Cr(1, i) - Lf_nom * Cf(1, i)) / (M_nom * v0) - v0). * yawrate_loss(1, i) +
                        Cf(1, i). * sas_loss(1, i) / M_nom)*T
    '''
    window_size = batch_data.size()[1]  # data number of columns
    if output.dim() == 1:
        #Vy_next = utils.bicycle_Vy2(output[0], output[1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])
        Yr_next = utils.bicycle_yr2(output[0], output[1], batch_data[0, 4], Beta, Yr, batch_data[0, 2])
        Beta_next = utils.bicycle_beta(output[0], output[1], batch_data[0, 4], Beta, Yr, batch_data[0, 2])
    else:
        #Vy_next = utils.bicycle_Vy2(output[0, 0], output[0, 1], batch_data[0, 4], Vy, Yr, batch_data[0, 2])
        Yr_next = utils.bicycle_yr2(output[0, 0], output[0, 1], batch_data[0, 4], Beta, Yr, batch_data[0, 2])
        Beta_next = utils.bicycle_beta(output[0, 0], output[0, 1], batch_data[0, 4], Beta, Yr, batch_data[0, 2])

    # print(Vy, label[1])
    loss_beta = torch.abs(Beta_next - label[1])
    loss_yr = torch.abs(Yr_next - label[0])
    '''
    print('vy error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Vy_next.item()) + str(' ') + \
                                       str(label[1].item()) + str(' ') + str(loss_vy.item()))

    assert abs(Vy_next.item()) < 100000, 'vy error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Vy_next.item()) + str(' ') + \
                                       str(label[1].item()) + str(' ') + str(loss_vy.item())
    assert abs(Yr_next.item()) < 100000, 'yr error ' + str(output[0, 0].item()) + str(' ') + str(output[0, 1].item()) + str(' ') + \
                                       str(batch_data[0, 4].item()) + str(' ') + str(Vy.item()) + str(' ') + str(Yr.item()) + \
                                       str(' ') + str(batch_data[0, 2].item()) + str(' ') + str(Yr_next.item()) + str(' ') + \
                                       str(label[0].item()) + str(' ') + str(loss_yr.item())
    '''
    # return (0.4 * loss_vy + 0.6 * loss_yr) / num_examples, Vy, Yr
    return loss_beta, loss_yr, Beta_next, Yr_next


def loss_lookup(batch, output):
    Cf_nom = 253000 * 0.8
    Cr_nom = 295000

    loss_function = nn.MSELoss()
    '''
    cf_lookup = torch.tensor(utils.lookup_table_cf(batch[:, 0, 1] * Cf_nom), dtype=torch.float64)
    cr_lookup = torch.tensor(utils.lookup_table_cr(batch[:, 0, 1] * Cr_nom), dtype=torch.float64)

    print(utils.lookup_table_cf(batch[1, 0, 1]), output[1, 0, 1])

    Cf = 1.2 * Cf_nom * output[:, 0, 0] + 0.05 * Cf_nom
    Cr = 1.2 * Cr_nom * output[:, 0, 1] + 0.05 * Cr_nom

    cf_loss = loss_function(Cf, cf_lookup)
    cr_loss = loss_function(Cr, cr_lookup)
    '''
    cf = torch.tensor(utils.lookup_table_cf(batch[:, 0, 1]), dtype=torch.float64)
    cr = torch.tensor(utils.lookup_table_cr(batch[:, 0, 1]), dtype=torch.float64)
    cf_loss = loss_function(cf, output[:, 0, 0])
    cr_loss = loss_function(cr, output[:, 0, 1])

    return cf_loss, cr_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def mse(outputs, labels):
    loss_func = nn.MSELoss()
    loss = loss_func(outputs, labels)
    return loss


def beta_loss(beta_loss, yr_loss):
    return beta_loss


def vy_loss(vy_loss, yr_loss):
    return vy_loss


def yr_loss(vy_loss, yr_loss):
    return yr_loss


def cf_loss(vy_loss, yr_loss):
    return vy_loss


def cr_loss(vy_loss, yr_loss):
    return yr_loss


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'Beta Loss': beta_loss,
    'Yawrate Loss': yr_loss,
    #'Vy Loss': vy_loss,
    #'Yr Loss': yr_loss
    #'MSE': mse,
    # could add more metrics such as accuracy for each token type
}
