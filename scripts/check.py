import torch
import torchvision.models as models
import torchvision
from torch import nn
from config import DROPOUT, DEVICE, MODEL_NAME, NUM_ROW, NUM_COLUMN, USE_DROP_OUT, USE_PRETRAIN

model = models.resnet50(pretrained=USE_PRETRAIN)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(DROPOUT),
                                     nn.Linear(num_features, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 16),
                                     nn.Sigmoid(),
                                     nn.Linear(16, 1))
print(model)