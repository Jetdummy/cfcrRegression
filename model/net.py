"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

import utils


class FourLayerModel(nn.Module):
    def __init__(self, params):
        self.num_channels = params.num_channels
        super(FourLayerModel, self).__init__()
        self.fc1 = nn.Linear(self.num_channels, 64)   # num_channels inputs -> 64 hidden units in the first layer
        self.fc2 = nn.Linear(64, 128) # 64 hidden units -> 128 hidden units in the second layer
        self.fc3 = nn.Linear(128, 64) # 128 hidden units -> 64 hidden units in the third layer
        self.fc4 = nn.Linear(64, 2)   # 64 hidden units -> 2 output in the fourth layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

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


def LSE_loss_fn(outputs, labels, batch_data):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    #Mf_nom = 904  # [Kg] 운전석에 1명 탔을때
    #Mr_nom = 619  # [Kg] 운전석에 1명 탔을때

    Mf_nom = 945  # [Kg] 운전석에 2명 탔을때
    Mr_nom = 728  # [Kg] 운전석에 2명탔을때

    M_nom = Mf_nom + Mr_nom  # [kg]

    Lf_nom = 2.645 * Mr_nom / M_nom  # [m]
    Lr_nom = 2.645 - Lf_nom  # [m]

    T = 0.01  # time step

    '''
    [batch_size][window_size][number of features]
    Cf = outputs[][0]
    Cr = outputs[][1]
    Vy_RT_loss = batch_data[][][7]
    yawrate_loss = batch_data[][][3]
    sas_loss = batch_data[][][5]
    Vy_RT_dot_loss = batch_data[][][6]
    v0(Vx) = batch_data[][][2]
    
    
    
    Vy_dot_est(1, i) = (-(Cf(1, i) + Cr(1, i)) / (M_nom * v0). * Vy_RT_loss(1, i) +
                        ((Lr_nom * Cr(1, i) - Lf_nom * Cf(1, i)) / (M_nom * v0) - v0). * yawrate_loss(1, i) +
                        Cf(1, i). * sas_loss(1, i) / M_nom)*T
    '''
    num_examples = batch_data.size()[0]  # batch size
    window_size = batch_data.size()[1]  # data number of columns

    # Vy를 출력해주기 위한 변수
    Vy = np.empty((0))

    loss = 0

    for i in range(num_examples):
        result = utils.bicycle_Vy(outputs[i][0][0], outputs[i][0][1], batch_data[i][-1][2], batch_data[i][-1][6],
                               batch_data[i][-1][3], batch_data[i][-1][5])
        '''
        result = (-(outputs[i][0][0] + outputs[i][0][1]) / (M_nom * batch_data[i][-1][2]) * batch_data[i][-1][6] +
                    ((Lr_nom * outputs[i][0][1] - Lf_nom * outputs[i][0][0]) / (M_nom * batch_data[i][-1][2]) - batch_data[i][-1][2]) * batch_data[i][-1][3] +
                    outputs[i][0][0] * batch_data[i][-1][5] / M_nom) * T
        '''
        loss += torch.mean((labels[i] - result) ** 2)
        result = result.data.cpu().numpy()
        Vy = np.concatenate((Vy, np.array([result])), axis=0)

    return loss, Vy


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
    print(outputs.shape)
    print(labels.shape)
    return mean_squared_error(outputs, labels)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    #'MSE': mse,
    # could add more metrics such as accuracy for each token type
}
