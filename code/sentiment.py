from dataset.data_loader import get_vico_dataloaders
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import Counter

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def extract(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class LSTM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_MLP, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, n_layers=2, dropout=0.3)
        self.mlp = MLP(hidden_dim*2, hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.mlp(x)
        return x

# data = get_vico_dataloaders(batch_size=4)
# dataloader = data['train']
# model = MLP(56, 256, 3).cuda()
# criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([432/113, 423/195, 432/115]).cuda())
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# epochs = 20

# best = 10**20
# for epoch in range(epochs):
#     losses = []
#     for batch in dataloader:
#         x, y = batch[1], batch[-2][-1]
#         # y: (B, 1) -> (B, T, 1)
        
#         new_x, new_y = [], []
#         # take frame by frame, skip 0 padded frames
#         B, T, D = x.shape
#         for i in range(B):
#             for j in range(T):
#                 if sum(x[i][j]) != 0:
#                     new_x.append(x[i][j])
#                     new_y.append(y[i])
#         x = torch.stack(new_x)
#         y = torch.stack(new_y)
#         x = x.cuda()
#         y = y.cuda()

#         y_pred = model(x)
#         loss = criterion(y_pred.reshape(-1, 3), y.reshape(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {np.mean(losses):.4f}')
#     if np.mean(losses) < best:
#         best = np.mean(losses)
#         torch.save(model.state_dict(), 'best_sentiment_model.pth')
    
model = MLP(56, 256, 3).cuda()
model.load_state_dict(torch.load('best_sentiment_model.pth'))

metadata_path = '../data/RLD_data.csv'
id2sentiment = {}
metadata = pd.read_csv(metadata_path).values
for i in range(len(metadata)):
    id2sentiment[metadata[i][1]] = metadata[i][0]
data_path = '../data/ELP_allfeaturespredicted_nodot0/'
all_exp_files = glob.glob(data_path + '**/exp.npy', recursive=True)
X, y, y_preds = [], [], []
for exp_file in tqdm(all_exp_files):
    exp = np.load(exp_file)
    pose = np.load(exp_file.replace('exp.npy', 'pose.npy'))
    current_vector = np.concatenate([pose, exp], axis=0)
    x = torch.FloatTensor(current_vector).cuda()
    y_pred = model.extract(x.unsqueeze(0))
    y.append(id2sentiment[exp_file.split('/')[-3]])
    X.append(y_pred.cpu().detach().numpy())
    y_pred = model(x.unsqueeze(0))
    y_pred = torch.softmax(y_pred, dim=-1)
    # y_pred = torch.argmax(y_pred)
    # y_preds.append(y_pred.item())

    # sort y_pred
    # proritize 2 > 0 > 1
    y_pred = y_pred.cpu().detach().numpy()[0]
    
    final_pred = None
    if y_pred[-1] > 0.03:
        final_pred = 2
    elif y_pred[0] > 0.41:
        final_pred = 0
    else:
        final_pred = 1
    y_preds.append(final_pred)

print(Counter(y), Counter(y_preds))

# X = np.array(X).reshape(-1, 256)
# # TSNE on the extracted features
# # tsne = TSNE(n_components=2, random_state=0)
# tsne = PCA(n_components=2)
# X_2d = tsne.fit_transform(X)
# # plot X_2d, color by y
# plt.figure()
# for i in range(len(X_2d)):
#     if y[i] == 'positive':
#         plt.scatter(X_2d[i][0], X_2d[i][1], c='r')
#     elif y[i] == 'negative':
#         plt.scatter(X_2d[i][0], X_2d[i][1], c='b')
#     else:
#         plt.scatter(X_2d[i][0], X_2d[i][1], c='g')
# plt.savefig('sentiment_tsne.png')