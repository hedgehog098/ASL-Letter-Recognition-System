import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import math

def load_data(train_txt_path, val_txt_path, prefix='shouyu/images'):
    train_list = []
    val_list = []
    
    with open(train_txt_path, 'r') as file:
        for line in file:
            path, label = line.strip().split(' ')
            train_list.append([f'{prefix}/{path}', int(label)])
    
    with open(val_txt_path, 'r') as file:
        for line in file:
            path, label = line.strip().split(' ')
            val_list.append([f'{prefix}/{path}', int(label)])
    
    print(f"训练集样本数量: {len(train_list)}")
    print(f"验证集样本数量: {len(val_list)}")
    
    return train_list, val_list

def visualize_dataset_distribution(train_list, val_list):
    category_mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
        10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q',
        19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'
    }
    
    train_counts = {i: 0 for i in range(29)}
    val_counts = {i: 0 for i in range(29)}
    
    for _, label in train_list:
        train_counts[label] += 1
    
    for _, label in val_list:
        val_counts[label] += 1
    
    train_counts_label = {category_mapping[k]: v for k, v in train_counts.items()}
    val_counts_label = {category_mapping[k]: v for k, v in val_counts.items()}
    
    plt.figure(figsize=(15, 8))
    bar_width = 0.4
    indices = list(range(len(train_counts_label)))
    
    train_values = list(train_counts_label.values())
    val_values = list(val_counts_label.values())
    labels = list(train_counts_label.keys())
    
    plt.bar([i - bar_width/2 for i in indices], train_values, bar_width, 
            label='Train', color='skyblue', alpha=0.8)
    plt.bar([i + bar_width/2 for i in indices], val_values, bar_width,
            label='Validation', color='orange', alpha=0.8)
    
    plt.xlabel('Label', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title('Dataset Distribution Visualization', fontsize=18, fontweight='bold')
    plt.xticks(indices, labels, rotation=45, ha='right')
    plt.legend()
    
    for i in indices:
        plt.text(i - bar_width/2, train_values[i] + 50, str(train_values[i]), 
                 ha='center', va='bottom', fontsize=8)
        plt.text(i + bar_width/2, val_values[i] + 50, str(val_values[i]), 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

class ASLDataset(Dataset):
    def __init__(self, data_list, transform=None, is_train=True):
        self.data_list = data_list
        self.is_train = is_train
        
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

def visualize_random_samples(dataset, num_samples=16):
    category_mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
        10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q',
        19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'
    }
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        image_np = image.numpy().transpose((1, 2, 0))
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        plt.subplot(4, 4, i + 1)
        plt.imshow(image_np)
        plt.title(f'Label: {category_mapping[label.item()]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

class ASLNet(nn.Module):
    def __init__(self, num_classes=29, pretrained=True):
        super(ASLNet, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-3
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 6
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*train_correct/train_total:.2f}%",
                'LR': f"{current_lr:.2e}"
            })
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*val_correct/val_total:.2f}%"
                })
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        overfit_gap = train_acc - val_acc
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"过拟合差距: {overfit_gap:.2f}%")
        print(f"Learning Rate: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss_avg,
                'overfit_gap': overfit_gap,
            }, 'model/best_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"验证准确率未提升 ({patience_counter}/{patience_limit})")
            
        if patience_counter >= patience_limit:
            print(f"早停触发！最佳验证准确率: {best_val_acc:.2f}%")
            break
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss_avg,
                'accuracy': val_acc,
                'overfit_gap': overfit_gap,
            }, f'model/checkpoint_epoch_{epoch+1}.pth')
    
    print(f"\n训练完成,最佳验证准确率: {best_val_acc:.2f}%")
    return history

def predict_image(model, image_path, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    return predicted_class.item(), probabilities.squeeze().cpu().numpy()

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-', marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Cosine Annealing Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    if len(epochs) > 0:
        overfit_gaps = [history['train_acc'][i] - history['val_acc'][i] for i in range(len(epochs))]
        axes[1, 1].plot(epochs, overfit_gaps, 'purple', marker='s', linewidth=2)
        axes[1, 1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10%差距线')
        axes[1, 1].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20%差距线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Overfitting Gap (%)')
        axes[1, 1].set_title('Train Acc - Val Acc (越小越好)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    train_list, val_list = load_data(
        train_txt_path='shouyu/images/train.txt',
        val_txt_path='shouyu/images/val.txt'
    )
    
    visualize_dataset_distribution(train_list, val_list)
    
    train_dataset = ASLDataset(train_list, is_train=True)
    val_dataset = ASLDataset(val_list, is_train=False)
    
    batch_size = 16
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    model = ASLNet(num_classes=29, pretrained=True).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数: {trainable_params:,}")
    print(f"总参数: {total_params:,}")
    print(f"训练比例: {trainable_params/total_params*100:.1f}%")
    
    
    num_epochs = 30
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    plot_training_history(history)
    
    if os.path.exists('model/best_model.pth'):
        checkpoint = torch.load('model/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n加载最佳模型 (epoch {checkpoint['epoch']+1}, acc: {checkpoint['val_acc']:.2f}%)")
        print(f"过拟合差距: {checkpoint['overfit_gap']:.2f}%")
    
    torch.save(model.state_dict(), 'model/final_model.pth')
    print("\nModel saved to 'model/final_model.pth'")
    
    print("\nTesting inference...")
    category_mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
        10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q',
        19: 'R', 20: 'S', 21: 'space', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'
    }
    
    model.load_state_dict(torch.load('model/final_model.pth', map_location=device))
    model.eval()
    
    print("\n在验证集上测试5个随机样本:")
    print("-" * 60)
    
    test_indices = np.random.choice(len(val_list), min(5, len(val_list)), replace=False)
    correct = 0
    
    for i, idx in enumerate(test_indices):
        test_image_path, true_label = val_list[idx]
        
        if os.path.exists(test_image_path):
            predicted_class, probabilities = predict_image(model, test_image_path, device)
            
            true_class_name = category_mapping[true_label]
            predicted_class_name = category_mapping[predicted_class]
            
            is_correct = predicted_class == true_label
            if is_correct:
                correct += 1
            
            print(f"样本 {i+1}:")
            print(f"  图片: {os.path.basename(test_image_path)}")
            print(f"  真实标签: {true_class_name}")
            print(f"  预测标签: {predicted_class_name}")
            print(f"  结果: {'正确' if is_correct else '错误'}")
            
            top1_confidence = probabilities[predicted_class] * 100
            print(f"  置信度: {top1_confidence:.2f}%")
            print("-" * 40)
    
    print(f"\n测试结果: {correct}/{len(test_indices)} 正确")
    print(f"测试准确率: {100*correct/len(test_indices):.2f}%")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    os.makedirs('model', exist_ok=True)
    main()