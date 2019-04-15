import torch
model = torch.load('./model/2019-04-11 23:36:30_model.pkl')
with open('para.txt', 'at') as f:
    for i in model.named_parameters():
        # print(i)
        f.write('{}\n'.format(i)) 