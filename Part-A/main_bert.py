from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt



def main():
    # Load data from CSV
    data = pd.read_csv('train_data.csv')
    label_map = {category: idx for idx, category in enumerate(sorted(data['label'].unique()))}
    data['label'] = data['label'].map(label_map)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Tokenize and encode sequences
    train_encodings = tokenizer(train_data['title'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    val_encodings = tokenizer(val_data['title'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Create TensorDataset
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_data['label'].tolist()))
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_data['label'].tolist()))

    # Define BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=21,)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-6)

    # Define batch size
    batch_size = 16

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transfer model to device
    model.to(device)

    # Training loop
    num_epochs = 10

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    best_val_accuracy = 0 

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        # Training loop
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

            loss.backward()
            optimizer.step()

        # Calculate training accuracy and loss
        train_accuracy = total_correct / total_examples
        train_accuracies.append(train_accuracy)
        train_losses.append(total_loss / len(train_loader))

        # Evaluation loop on validation data
        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation accuracy and loss
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))

        if val_accuracy > best_val_accuracy:
           print(f'Epoch {epoch + 1} save model')
           best_val_accuracy = val_accuracy
           best_epoch = epoch + 1
           # Save the model at the best validation accuracy
           model_save_path = f"bert_model_tt_bc16_lr6.pth"
           torch.save({
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'label_map': label_map,
               'tokenizer': tokenizer,
               'best_val_accuracy': best_val_accuracy,
               'best_epoch': best_epoch,
           }, model_save_path)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]}, Training Accuracy: {train_accuracies[-1]}, Validation Loss: {val_losses[-1]}, Validation Accuracy: {val_accuracies[-1]}')

    # Plot training and validation accuracy and loss per epoch
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()