from typing import Optional, Type, List
from dataclasses import dataclass, replace, field

import torch as t
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import wandb

from tqdm import tqdm

from resnet import ResNet34

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

IMAGE_SIZE = 32

cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

imgs = [item[0] for item in cifar]
imgs = t.stack(imgs, dim=0)

std, mean = t.std_mean(imgs, dim=(0, 2, 3))


print(f"{mean=}, {std=}")


IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=mean, std=std),
])

def get_cifar(subset: int):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)

    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        # Still return the whole test set
        # cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset


@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10


@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    wandb_project: Optional[str] = 'resnet34-scaling-laws'
    wandb_name: Optional[str] = None

class ResNetTrainerWandb:
  def __init__(self, args: ResNetTrainingArgsWandb):
    self.args = args
    self.model = ResNet34().to(device)
    self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.learning_rate)
    self.trainset, self.testset = get_cifar(subset=args.subset)

  def to_device(self, *args):
    return [x.to(device) for x in args]

  def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
    imgs, labels = self.to_device(imgs, labels)
    logits = self.model(imgs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    return loss

  @t.inference_mode()
  def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
    imgs, labels = self.to_device(imgs, labels)
    logits = self.model(imgs)
    return (logits.argmax(dim=1) == labels).sum()

  def train(self):
    wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
    wandb.watch(self.model, log='all', log_freq=100)

    step = 0

    for epoch in range(self.args.epochs):

      # Load data
      train_dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
      val_dataloader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
      progress_bar = tqdm(total=len(train_dataloader))

      # Training loop (includes updating progress bar, and logging loss)
      self.model.train()
      for imgs, labels in train_dataloader:
        loss = self.training_step(imgs, labels)
        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}")

        wandb.log({"loss": loss.item()}, step)

        step += 1

      # Compute accuracy by summing n_correct over all batches, and dividing by number of items
      self.model.eval()
      accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in val_dataloader) / len(self.testset)

      # Update progress bar description to include accuracy, and log accuracy
      progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")
      wandb.log({"accuracy": accuracy.item()}, step)


def fac(arg: List):
    return field(default_factory=lambda: arg)

@dataclass
class ResNetParams(ResNetTrainingArgsWandb):
    n_blocks_per_group: List[int] = fac([3, 4, 6, 3])
    out_features_per_group: List[int] = fac([64, 128, 256, 512])
    first_strides_per_group: List[int] = fac([1, 2, 2, 2])
    n_classes: int = 10


class ResNetTrainerWandbSweeps(ResNetTrainerWandb):
    '''
    New training class made specifically for hyperparameter sweeps, which overrides the values in
    `args` with those in `wandb.config` before defining model/optimizer/datasets.
    '''
    def __init__(self, args: ResNetParams):

        # Initialize
        wandb.init(name=args.wandb_name)

        # Update args with the values in wandb.config
        self.args = replace(args, **wandb.config)

        # Perform the previous steps (initialize model & other important objects)
        self.model = ResNet34(
            self.args.n_blocks_per_group,
            self.args.out_features_per_group,
            self.args.first_strides_per_group,
            self.args.n_classes,
            ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=self.args.subset)
        self.step = 0
        wandb.watch(self.model, log="all", log_freq=20)

if __name__ == '__main__':
    sweep_config = dict(
        method="random",
        metric={"goal": "maximize", "name": "accuracy"},
        parameters={
            "batch_size": {"values": [64, 128]},
            "learning_rate": {"distribution": "log_uniform_values", "min": 0.0007 / 10, "max": 0.0007 * 10},
            "out_features_per_group": {"values": [[64, 128, 256, 512], [32, 64, 128, 256], [128, 256, 512, 1024]]},
            "n_blocks_per_group": {"values": [[3,4,6,3], [2, 2, 2, 2], [5, 6, 8, 5]]},
        }
    )


    args = ResNetParams()
    args.epochs = 10
    args.learning_rate = 0.0007
    args.wandb_project = "resnet_10_epoch"

    def train():
        trainer = ResNetTrainerWandbSweeps(args)
        trainer.train()

    sweep_id = wandb.sweep(sweep=sweep_config, project='resnet_10_epoch_2')
    wandb.agent(sweep_id=sweep_id, function=train, count=12)
    wandb.finish()
