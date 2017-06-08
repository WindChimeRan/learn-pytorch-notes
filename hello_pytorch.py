import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

batch_size = 100
learning_rate = 1e-3
num_epoches = 20

train_dataset = datasets.MNIST(root='./mnist', train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist', train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classifier = nn.Sequential(
    nn.Linear(784,256),
    nn.LeakyReLU(0.1),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.1),
    nn.Linear(128, 10),
    nn.Sigmoid()
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(),lr=1e-3)
classifier.cuda()

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        b, c, h, w = img.size()

        img = img.view(batch_size,-1)
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        out = classifier(img)


        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch+1, num_epoches,
                running_loss/(batch_size*i),
                running_acc/(batch_size*i)
            ))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch+1,
        running_loss/(len(train_dataset)),
        running_acc/(len(train_dataset))
    ))
    classifier.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        b, c, h, w = img.size()

        img = img.view(batch_size, -1)

        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()

        out = classifier(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        (eval_acc + 0.0) / (len(test_dataset))
    ))

    print(' ')

torch.save(classifier.state_dict(), './model/model.pth')