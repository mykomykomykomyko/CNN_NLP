import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

text_field = torchtext.data.Field(lower=True)
label_field = torchtext.data.Field(sequential=False)
train_data, test_data = torchtext.datasets.IMDB.splits(text_field, label_field) 
text_field.build_vocab(train_data)
label_field.build_vocab(train_data)
train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, test_data), batch_size=32, device=-1)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, kernel_dim=100, kernel_sizes=(3,4,5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, 
                                              out_channels=kernel_dim, 
                                              kernel_size=(kernel_size, embedding_dim)) 
                                    for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_dim)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())

        
    def forward(self, text):
        text = text.permute(1, 0)  
        embedded = self.embedding(text) 
        embedded = embedded.unsqueeze(1) 
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


    for epoch in range(num_epochs):
        for batch in train_iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)
            loss = loss_function(predictions, batch.label)
            loss.backward()
            optimizer.step()

true_Attempts, total_Attempts = 0
with torch.no_grad():
    for batch in test_iterator:
        predictions = model(batch.text).squeeze(1)
        predicted = (torch.sigmoid(predictions) > 0.5).long()
        total_Attempts += batch.label.size(0)
        true_Attempts += (predicted == batch.label).sum().item()
factor_Success = true_Attempts / total_Attempts
print(f'Success: {factor_Success:.2f}')

