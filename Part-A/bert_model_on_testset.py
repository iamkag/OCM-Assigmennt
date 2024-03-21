from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd

def load_model(model_save_path):
    checkpoint = torch.load(model_save_path)
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=21)
    model.load_state_dict(checkpoint['model_state_dict'])
    label_map = checkpoint['label_map']
    tokenizer = checkpoint['tokenizer']
    return model, label_map, tokenizer

def test_model(model, tokenizer, test_loader):

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

    return predictions, ground_truth

def calculate_f1_per_class(predictions, ground_truth, label_map):
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average=None)
    f1_per_class = {label: f1_score for label, f1_score in zip(range(len(label_map)), f1)}
    return f1_per_class

def main():
    # Load the saved model
    model, label_map, tokenizer = load_model("bert_model_tt_bc16_lr6.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load test data
    test_data = pd.read_csv('unseen_test_data.csv')
    
    # Map labels in test data using the same label map from training
    test_data['label'] = test_data['label'].map(label_map)

    # Tokenize and encode test sequences
    test_encodings = tokenizer(test_data['title'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
   
    # Create TensorDataset
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_data['label'].tolist()))
  
    # Create DataLoader
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Test the model
    predictions, ground_truth = test_model(model, tokenizer, test_loader)

    # Test the model
    predictions, ground_truth = test_model(model, tokenizer, test_loader)

    # Calculate F1 score per class
    f1_per_class = calculate_f1_per_class(predictions, ground_truth, label_map)

    # Print F1 score per class
    key_list = list(label_map.keys())
    value_list = list(label_map.values())
    for label, scores in f1_per_class.items():
        print(f"Class: {key_list[value_list.index(label)]}-{label}")
        print(f"F1 Score: {scores}")
        print()

if __name__ == "__main__":
    main()