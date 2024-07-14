import torch
import torch.nn as nn
from torchinfo import summary


# Tested Rationality        [ ]
# Confirmed by third party  [ ]
# Ran tests                 [ ]


class Madry(nn.Module):
    def __init__(self):
        super(Madry, self).__init__()
        self.conv1x = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding="same"),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2x = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding="same"),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(),
        )
        self.feat_dim = 1024
        self.fc2 = nn.Linear(1024, 10)

        self.grad_layer = [self.pool1, self.pool2]
        # nn.Sequential(self.conv2x[0], self.conv2x[1], self.pool2)

    def get_softmax_weights(self):
        return self.fc2.weight.data

    def forward(self, x):
        x = self.conv1x(x)
        x = self.pool1(x)

        x = self.conv2x(x)
        x = self.pool2(x)

        x = self.flatten(x)

        feats = self.fc1(x)
        logits = self.fc2(feats)
        return feats, logits

    def freeze_FE(self):
        for name, param in self.named_parameters():
            if "fc2" not in name:
                param.requires_grad = False

    def freeze_last_layer(self):
        for name, param in self.named_parameters():
            if "fc2" in name:
                param.requires_grad = False


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Madry().to(device)
    print(summary(model, (1, 1, 28, 28)))
