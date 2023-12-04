import torch
import torchvision.models as models
import torchvision
from torch import nn
from config import DROPOUT, DEVICE, MODEL_NAME, NUM_ROW, NUM_COLUMN, USE_DROP_OUT, USE_PRETRAIN


def create_model(new_model=USE_PRETRAIN):
    if MODEL_NAME == "resnet50" or MODEL_NAME == "resnet18" or MODEL_NAME == "resnet34" or MODEL_NAME == "resnet101" or MODEL_NAME == "resnet152":
        # Resnet50
        if MODEL_NAME == "resnet50":
            model = models.resnet50(pretrained=new_model)
        elif MODEL_NAME == "resnet34":
            model = models.resnet34(pretrained=new_model)
        elif MODEL_NAME == "resnet101":
            model = models.resnet101(pretrained=new_model)
        elif MODEL_NAME == "resnet152":
            model = models.resnet152(pretrained=new_model)
        else:
            model = models.resnet18(pretrained=new_model)

        num_features = model.fc.in_features

        # model.fc = nn.Linear(num_features, 1)
        if USE_DROP_OUT:
            model.fc = nn.Sequential(nn.Dropout(DROPOUT),
                                     nn.Linear(num_features, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 16),
                                     nn.Sigmoid(),
                                     nn.Linear(16, 1))
        else:
            model.fc = nn.Sequential(nn.Linear(num_features, 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, 16),
                                     nn.Sigmoid(),
                                     nn.Linear(16, 1))

    elif MODEL_NAME == "efficientnet_b7":
        # Efficientnet_b7
        model = models.efficientnet_b7(pretrained=new_model)
        num_features = model.classifier[1].in_features
        if USE_DROP_OUT:
            model.classifier = nn.Sequential(nn.Dropout(DROPOUT),
                                             nn.Linear(num_features, 256),
                                             nn.LeakyReLU(),
                                             nn.Linear(256, 32),
                                             nn.Sigmoid(),
                                             nn.Linear(32, 1))
        else:
            model.classifier = nn.Sequential(nn.Linear(num_features, 512),
                                             nn.LeakyReLU(),
                                             nn.Linear(512, 256),
                                             nn.LeakyReLU(),
                                             nn.Linear(256, 64),
                                             nn.Sigmoid(),
                                             nn.Linear(64, 1))

    elif MODEL_NAME == "inceptionv3":
        # Inceptionv3
        model = models.inception_v3(pretrained=new_model)

        num_features = model.fc.in_features
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        model.fc = nn.Sequential(nn.Linear(num_features, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 1))
    else:
        Exception(f"{MODEL_NAME} is an invalid model's name")

    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.dropout = nn.Dropout(DROPOUT)

    return model


if __name__ == '__main__':
    model = create_model().to(DEVICE)
    print(model)
