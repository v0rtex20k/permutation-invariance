import torch
import torchvision
from abc import ABC, abstractmethod


class DatasetConfig(ABC):
    """Abstract base class for dataset configurations."""
    
    @abstractmethod
    def get_transforms(self):
        """Return the transforms for this dataset."""
        pass
    
    @abstractmethod
    def load_datasets(self, data_dir):
        """Load and return train and validation datasets."""
        pass
    
    @abstractmethod
    def get_model(self, architecture, pretrained=True):
        """Return the model for this dataset."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self):
        """Return the number of classes in the dataset."""
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the dataset."""
        pass


class ImageNetConfig(DatasetConfig):
    """Configuration for ImageNet-1k dataset."""
    
    def get_transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(256, 256)),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load_datasets(self, data_dir):
        transform = self.get_transforms()
        train_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
        val_dataset = torchvision.datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
        return train_dataset, val_dataset
    
    def get_model(self, architecture, pretrained=True):
        if architecture == "resnet18":
            if pretrained:
                return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                return torchvision.models.resnet18(weights=None, num_classes=self.num_classes)
        elif architecture == "resnet50":
            if pretrained:
                return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                return torchvision.models.resnet50(weights=None, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported architecture for ImageNet: {architecture}")
    
    @property
    def num_classes(self):
        return 1000
    
    @property
    def name(self):
        return "imagenet"


class CIFAR10Config(DatasetConfig):
    """Configuration for CIFAR-10 dataset."""
    
    def get_transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    
    def load_datasets(self, data_dir):
        transform = self.get_transforms()
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, val_dataset
    
    def get_model(self, architecture, pretrained=True):
        if architecture == "resnet18":
            model = torchvision.models.resnet18(weights=None, num_classes=self.num_classes)
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
            return model
        elif architecture == "resnet50":
            model = torchvision.models.resnet50(weights=None, num_classes=self.num_classes)
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
            return model
        else:
            raise ValueError(f"Unsupported architecture for CIFAR-10: {architecture}")
    
    @property
    def num_classes(self):
        return 10
    
    @property
    def name(self):
        return "cifar10"


def get_dataset_config(dataset_name):
    """Factory function to get the appropriate dataset configuration."""
    configs = {
        "imagenet": ImageNetConfig(),
        "cifar10": CIFAR10Config(),
    }
    
    if dataset_name.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(configs.keys())}")
    
    return configs[dataset_name.lower()]
