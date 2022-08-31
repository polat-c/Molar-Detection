import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision.models.resnet import model_urls
import segmentation_models_pytorch as smp
model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')



class ConvNet(nn.Module):
    def __init__(self, n_class, img_width):
        super(ConvNet, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # FC Layers
        width = int(img_width / 4 - 3)
        self.fc1 = nn.Linear(16 * width * width, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

        # Drop Layers
        self.dropout_conv = nn.Dropout2d(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout_conv(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(F.relu(self.conv2(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        # x = self.dropout_fc(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer


class ResNetXIn(nn.Module):
    '''Resnet implementation accepting inputs of varied sizes'''

    def __init__(self, size, pretrain, in_channels=1):
        super(ResNetXIn, self).__init__()

        # bring resnet
        if size == 18:
            self.model = torchvision.models.resnet18(pretrained=pretrain)
        if size == 34:
            self.model = torchvision.models.resnet34(pretrained=pretrain)
        if size == 50:
            self.model = torchvision.models.resnet50(pretrained=pretrain)
        if size == 101:
            self.model = torchvision.models.resnet101(pretrained=pretrain)
        if size == 152:
            self.model = torchvision.models.resnet152(pretrained=pretrain)

        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer


def criterion(x, labels):
    c_entropy = F.cross_entropy(x, labels)

    return c_entropy

#################################################

class DetectionConvNet(nn.Module): # just try out vanilla first
    def __init__(self, img_width):
        super(DetectionConvNet, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # FC Layers
        width = int(img_width / 4 - 3)
        self.fc1 = nn.Linear(16 * width * width, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) # changed output size from n_classes to 2, since this is regression problem for ROI center

        # Drop Layers
        self.dropout_conv = nn.Dropout2d(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout_conv(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(F.relu(self.conv2(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        # x = self.dropout_fc(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

class DetectionResNetXIn(nn.Module):
    '''Resnet implementation accepting inputs of varied sizes'''

    def __init__(self, size, pretrain, in_channels=1):
        super(DetectionResNetXIn, self).__init__()

        # bring resnet
        if size == 18:
            self.model = torchvision.models.resnet18(pretrained=pretrain)
        if size == 34:
            self.model = torchvision.models.resnet34(pretrained=pretrain)
        if size == 50:
            self.model = torchvision.models.resnet50(pretrained=pretrain)
        if size == 101:
            self.model = torchvision.models.resnet101(pretrained=pretrain)
        if size == 152:
            self.model = torchvision.models.resnet152(pretrained=pretrain)

        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

def detection_criterion_regressor(x, target): # first try: naive criterion (mse)
    mse_loss = F.mse_loss(x, target)

    return mse_loss


class DetectionUnet(nn.Module): # just try out vanilla first
    def __init__(self):
        super(DetectionUnet, self).__init__()

        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
        #DEVICE = 'cuda'

        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=1,
            activation=ACTIVATION,
        )
        self.model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # changing input channel amount
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    def forward(self, x):
        return self.model(x)

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

def detection_criterion_classifier(x, target): # first try: naive criterion (mse)
    Dice_loss = smp.utils.losses.DiceLoss()
    loss = Dice_loss.forward(x, target)

    return loss


