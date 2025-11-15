import argparse
import os
import copy
import random
import itertools
import numpy as np
# PyTorch
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torchvision
# Importing our custom module(s)
import permutations
import utils

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    
def evaluate(model, criterion, dataloader):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"acc1": 0.0, "acc5": 0.0, "loss": 0.0, "nll": 0.0}
            
    with torch.no_grad():
        #for images, labels in dataloader:
        for images, labels in itertools.islice(iter(dataloader), 10):
            
            batch_size = len(images)
                        
            if device.type == "cuda":
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, labels)
            
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            metrics["acc1"] += (batch_size / dataset_size) * acc1.item()
            metrics["acc5"] += (batch_size / dataset_size) * acc5.item()
            metrics["loss"] += (batch_size / dataset_size) * loss.item()
    
    return metrics

def interpolate_state_dicts(state_dict1, state_dict2, num_alphas=11):
    
    state_dicts = [{} for _ in range(num_alphas)]
    alphas = np.linspace(start=0, stop=1.0, num=num_alphas)
    
    for idx, alpha in enumerate(alphas):
        
        for key in state_dict1.keys():
            
            state_dicts[idx][key] = (alpha * copy.deepcopy(state_dict1[key].detach())) + ((1 - alpha) * copy.deepcopy(state_dict2[key].detach()))
            
            if "running_var" in key and alpha > 1.0:
                state_dicts[idx][key] = copy.deepcopy(state_dict1[key].detach())
            elif "running_var" in key and alpha < 0.0:
                state_dicts[idx][key] = copy.deepcopy(state_dict2[key].detach())
                
    return state_dicts

def permute_one_neuron(model):
    
    layer_name = random.choice(["layer1", "layer2", "layer3", "layer4"])
    layer = getattr(model, layer_name)
    block = random.choice(list(layer))
    block_layers = random.choice([["conv1", "bn1", "conv2"], ["conv2", "bn2", "conv3"]])

    conv_a = getattr(block, block_layers[0])
    bn = getattr(block, block_layers[1])
    conv_b = getattr(block, block_layers[2])

    C = conv_a.out_channels
    perm = torch.arange(C)
    i, j = random.sample(range(C), 2)
    temp_idx = copy.deepcopy(perm[i])
    perm[i] = copy.deepcopy(perm[j])
    perm[j] = temp_idx

    permutations.permute_conv2d_out_channels(conv_a, perm)
    permutations.permute_batchnorm2d(bn, perm)
    permutations.permute_conv2d_in_channels(conv_b, perm)

    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="main_imagenet1k.py")
    parser.add_argument("--batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--experiments_path", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--imagenet1k_dir", default="", help="Directory to ImageNet-1k (default: \"\")", type=str)
    parser.add_argument("--num_alphas", default=11, help="Number of alphas (default: 11)", type=int)
    parser.add_argument("--num_batches", default=11, help="Number of batches (default: 10)", type=int)
    parser.add_argument("--num_perms", default=1, help="Number of permutations (default: 1)", type=int)
    parser.add_argument("--num_workers", default=16, help="Number of workers (default: 16)", type=int)
    parser.add_argument("--random_state", default=42, help="Random state (default: 42)", type=int)
    args = parser.parse_args()
            
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    
    os.makedirs(os.path.dirname(args.experiments_path), exist_ok=True)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root=f"{args.imagenet1k_dir}/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=f"{args.imagenet1k_dir}/val", transform=transform)
    
    train_indices = torch.randperm(len(train_dataset))[:(args.num_batches * args.batch_size)].view(args.num_batches, args.batch_size)
    val_indices = torch.randperm(len(val_dataset))[:(args.num_batches * args.batch_size)].view(args.num_batches, args.batch_size)
            
    model1 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model2 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    for _ in range(args.num_perms):
        model2 = permute_one_neuron(model2)
        
    params1 = torch.nn.utils.parameters_to_vector(model1.parameters())
    params2 = torch.nn.utils.parameters_to_vector(model2.parameters())

    l2_norm = ((params1 - params2)**2).sum()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model1.to(device)
    model2.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    state_dicts = interpolate_state_dicts(model1.state_dict(), model2.state_dict(), args.num_alphas)
    
    train_losses = torch.zeros(size=(args.num_batches, args.num_alphas))
    train_acc1s = torch.zeros(size=(args.num_batches, args.num_alphas))
    train_acc5s = torch.zeros(size=(args.num_batches, args.num_alphas))
    val_losses = torch.zeros(size=(args.num_batches, args.num_alphas))
    val_acc1s = torch.zeros(size=(args.num_batches, args.num_alphas))
    val_acc5s = torch.zeros(size=(args.num_batches, args.num_alphas))
    
    for batch_idx in range(args.num_batches):
        
        train_batch = torch.utils.data.Subset(train_dataset, train_indices[batch_idx])
        val_batch = torch.utils.data.Subset(val_dataset, val_indices[batch_idx])
        
        train_dataloader = torch.utils.data.DataLoader(train_batch, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
        val_dataloader = torch.utils.data.DataLoader(val_batch, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
        
        for state_dict_idx, state_dict in enumerate(state_dicts):
                
            model1.load_state_dict(state_dict)
            train_metrics = evaluate(model1, criterion, train_dataloader)
            train_losses[batch_idx, state_dict_idx] = train_metrics["loss"]
            train_acc1s[batch_idx, state_dict_idx] = train_metrics["acc1"]
            train_acc5s[batch_idx, state_dict_idx] = train_metrics["acc5"]
            val_metrics = evaluate(model1, criterion, val_dataloader)
            val_losses[batch_idx, state_dict_idx] = val_metrics["loss"]
            val_acc1s[batch_idx, state_dict_idx] = val_metrics["acc1"]
            val_acc5s[batch_idx, state_dict_idx] = val_metrics["acc5"]
                
    torch.save({
        "l2_norm": l2_norm,
        "train_losses": train_losses,
        "train_acc1s": train_acc1s,
        "train_acc5s": train_acc5s,
        "val_losses": val_losses,
        "val_acc1s": val_acc1s,
        "val_acc5s": val_acc5s,
    }, args.experiments_path)
