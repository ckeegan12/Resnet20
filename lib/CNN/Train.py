import torch.nn as nn
import torch.optim as optim

class model_training:
    def __init__(self, model, train_loader, epochs):
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, device):
        print(f"Start Training for {self.epochs} epochs")
        train_loss_list = []
        lr = 0.1
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.5e-4)

        for epoch in range(self.epochs):
            epoch_losses = []
            self.model.train()

            if epoch // 50:
                lr = lr * 0.1**(epoch // 50 - 1) 
                optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.5e-4)

            correct = 0
            total = 0

            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # back pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Calculate average for this epoch only
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            train_loss_list.append(epoch_loss)  
            print(f'Epoch [{epoch+1}/{self.epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {100. * (correct/total):.2f}')
            
        return train_loss_list
