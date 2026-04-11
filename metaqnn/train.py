import torch
from tqdm import tqdm
from metaqnn.train_config import DEVICE


def validate(model, val_loader):
    model.eval()
    num_correct = 0
    num_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(inputs)

            pred_classes = preds.argmax(dim=1)

            num_correct += (pred_classes == labels).sum().item()
            num_total += len(labels)

    accuracy = num_correct / num_total

    print(f'Model accuracy: {accuracy}')

    return accuracy


def train(model, num_epochs, train_loader, val_loader, loss_func, optimizer, scheduler):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model on first epoch
        if epoch == 0:
            accuracy = validate(model, val_loader)

            # Model is worse than random chance
            # TODO: restart it a few times if this happens
            if accuracy <= 0.1:
                return accuracy

        # Step scheduler
        if scheduler:
            scheduler.step()

    # Evaluate model
    accuracy = validate(model, val_loader)

    return accuracy
