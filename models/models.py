import torch
import pytorch_lightning as pl

from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, num_classes = 10, dropout = 0.5, lr = 0.01):
        super().__init__()
        self.alexnet = AlexNet(num_classes, dropout)
        self.lr = lr
        
    def forward(self, x: torch.Tensor):
        output = self.alexnet(x)
        return output.argmax(dim=-1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)
        
        self.log('train_loss', loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)
        acc = (out.argmax(dim=-1) == y).sum() / y.size(0)
        
        self.log('val_loss', loss.item())
        self.log('val_acc', acc.item())
        
        return loss, acc
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)
        pred = out.argmax(dim=-1)
        acc = (pred == y).sum() / y.size(0)

        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())

        return loss, acc