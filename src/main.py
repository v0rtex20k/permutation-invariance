import argparse
import os
import copy
import random
import itertools
import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torchvision
import permutations
import utils
import dataset_configs

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
    
def evaluate(model, criterion, dataloader, num_classes=1000):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    
    if num_classes >= 5:
        metrics = {"acc1": 0.0, "acc5": 0.0, "loss": 0.0, "nll": 0.0}
    else:
        metrics = {"acc1": 0.0, "loss": 0.0, "nll": 0.0}
            
    with torch.no_grad():
        for images, labels in itertools.islice(iter(dataloader), 10):
            
            batch_size = len(images)
                        
            if device.type == "cuda":
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            loss = criterion(logits, labels)
            
            if num_classes >= 5:
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                metrics["acc1"] += (batch_size / dataset_size) * acc1.item()
                metrics["acc5"] += (batch_size / dataset_size) * acc5.item()
            else:
                acc1 = accuracy(logits, labels, topk=(1,))[0]
                metrics["acc1"] += (batch_size / dataset_size) * acc1.item()
            
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

def permute_one_neuron(model, is_resnet18=False):
    
    layer_name = random.choice(["layer1", "layer2", "layer3", "layer4"])
    layer = getattr(model, layer_name)
    block = random.choice(list(layer))
    
    if is_resnet18:
        block_layers = ["conv1", "bn1", "conv2"]
    else:
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

def permute_full_layer(model, is_resnet18=False):
    """Apply a full random permutation to ALL intermediate channels in a random block."""
    
    layer_name = random.choice(["layer1", "layer2", "layer3", "layer4"])
    layer = getattr(model, layer_name)
    block = random.choice(list(layer))
    
    if is_resnet18:
        perm = torch.randperm(block.conv1.out_channels)
        permutations.permute_conv2d_out_channels(block.conv1, perm)
        permutations.permute_batchnorm2d(block.bn1, perm)
        permutations.permute_conv2d_in_channels(block.conv2, perm)
    else:
        perm1 = torch.randperm(block.conv1.out_channels)
        permutations.permute_conv2d_out_channels(block.conv1, perm1)
        permutations.permute_batchnorm2d(block.bn1, perm1)
        permutations.permute_conv2d_in_channels(block.conv2, perm1)
        
        perm2 = torch.randperm(block.conv2.out_channels)
        permutations.permute_conv2d_out_channels(block.conv2, perm2)
        permutations.permute_batchnorm2d(block.bn2, perm2)
        permutations.permute_conv2d_in_channels(block.conv3, perm2)

    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="main.py - Permutation Invariance Experiments")
    parser.add_argument("--dataset", default="imagenet", help="Dataset to use: imagenet or cifar10 (default: imagenet)", type=str)
    parser.add_argument("--data_dir", default="", help="Directory to dataset (default: \"\")", type=str)
    parser.add_argument("--batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--experiments_path", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--num_alphas", default=11, help="Number of alphas (default: 11)", type=int)
    parser.add_argument("--num_batches", default=11, help="Number of batches (default: 10)", type=int)
    parser.add_argument("--num_perms", default=1, help="Number of permutations (default: 1)", type=int)
    parser.add_argument("--num_workers", default=16, help="Number of workers (default: 16)", type=int)
    parser.add_argument("--random_state", default=42, help="Random state (default: 42)", type=int)
    parser.add_argument("--model", default="resnet50", help="Model architecture: resnet18 or resnet50 (default: resnet50)", type=str)
    parser.add_argument("--full_perm", action="store_true", help="Use full layer permutation instead of single neuron swaps")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained models (only for ImageNet)")
    args = parser.parse_args()
            
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    
    os.makedirs(os.path.dirname(args.experiments_path), exist_ok=True)
    
    dataset_config = dataset_configs.get_dataset_config(args.dataset)
    print(f"Using dataset: {dataset_config.name}")
    print(f"Number of classes: {dataset_config.num_classes}")
    
    train_dataset, val_dataset = dataset_config.load_datasets(args.data_dir)
    
    train_indices = torch.randperm(len(train_dataset))[:(args.num_batches * args.batch_size)].view(args.num_batches, args.batch_size)
    val_indices = torch.randperm(len(val_dataset))[:(args.num_batches * args.batch_size)].view(args.num_batches, args.batch_size)
            
    is_resnet18 = (args.model == "resnet18")
    
    use_pretrained = args.pretrained and (args.dataset == "imagenet")
    if not use_pretrained and args.pretrained:
        print(f"Warning: Pretrained models not available for {args.dataset}, using random initialization")
    
    model1 = dataset_config.get_model(args.model, pretrained=use_pretrained)
    model2 = dataset_config.get_model(args.model, pretrained=use_pretrained)
    print(f"Using {args.model}...")
    print(f"Pretrained: {use_pretrained}")

    for _ in range(args.num_perms):
        if args.full_perm:
            model2 = permute_full_layer(model2, is_resnet18)
        else:
            model2 = permute_one_neuron(model2, is_resnet18)
        
    params1 = torch.nn.utils.parameters_to_vector(model1.parameters())
    params2 = torch.nn.utils.parameters_to_vector(model2.parameters())

    l2_norm = ((params1 - params2)**2).sum()
    print(f"L2 norm between models: {l2_norm.item():.6f}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model1.to(device)
    model2.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    state_dicts = interpolate_state_dicts(model1.state_dict(), model2.state_dict(), args.num_alphas)
    
    train_losses = torch.zeros(size=(args.num_batches, args.num_alphas))
    train_acc1s = torch.zeros(size=(args.num_batches, args.num_alphas))
    val_losses = torch.zeros(size=(args.num_batches, args.num_alphas))
    val_acc1s = torch.zeros(size=(args.num_batches, args.num_alphas))
    
    if dataset_config.num_classes >= 5:
        train_acc5s = torch.zeros(size=(args.num_batches, args.num_alphas))
        val_acc5s = torch.zeros(size=(args.num_batches, args.num_alphas))
    
    for batch_idx in range(args.num_batches):
        
        train_batch = torch.utils.data.Subset(train_dataset, train_indices[batch_idx])
        val_batch = torch.utils.data.Subset(val_dataset, val_indices[batch_idx])
        
        train_dataloader = torch.utils.data.DataLoader(train_batch, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
        val_dataloader = torch.utils.data.DataLoader(val_batch, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
        
        for state_dict_idx, state_dict in enumerate(state_dicts):
                
            model1.load_state_dict(state_dict)
            train_metrics = evaluate(model1, criterion, train_dataloader, num_classes=dataset_config.num_classes)
            train_losses[batch_idx, state_dict_idx] = train_metrics["loss"]
            train_acc1s[batch_idx, state_dict_idx] = train_metrics["acc1"]
            
            val_metrics = evaluate(model1, criterion, val_dataloader, num_classes=dataset_config.num_classes)
            val_losses[batch_idx, state_dict_idx] = val_metrics["loss"]
            val_acc1s[batch_idx, state_dict_idx] = val_metrics["acc1"]
            
            if dataset_config.num_classes >= 5:
                train_acc5s[batch_idx, state_dict_idx] = train_metrics["acc5"]
                val_acc5s[batch_idx, state_dict_idx] = val_metrics["acc5"]
    
    results = {
        "l2_norm": l2_norm,
        "train_losses": train_losses,
        "train_acc1s": train_acc1s,
        "val_losses": val_losses,
        "val_acc1s": val_acc1s,
    }
    
    if dataset_config.num_classes >= 5:
        results["train_acc5s"] = train_acc5s
        results["val_acc5s"] = val_acc5s
    
    torch.save(results, args.experiments_path)
    print(f"Results saved to: {args.experiments_path}")
