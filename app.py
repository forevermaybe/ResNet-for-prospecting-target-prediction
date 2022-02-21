import torch.nn as nn
import torch
import dataset.traindataloader as traindataloader
import dataset.testdataloader as testdataloader
from misc import AverageMeter
import time
from model import resnet


def accuracy(output, target, topk=(1,)):
    """Computes the precision for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k1 = correct[:k]
            correct_k1.contiguous()
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(batch_size):
    model = resnet.resnet50(num_classes=2)
    model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    train_dataloader = traindataloader.get_train_loader('data', batch_size)
    lr_decay_epoch = [30, 60, 90, 120]
    epochs = 120
    lr_decay_count = 0
    lr_decay = 0.5
    for epoch in range(epochs):
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        if epoch == lr_decay_epoch[lr_decay_count]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay
            lr_decay_count += 1
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            inputs = inputs.clone().detach().float()
            output= model(inputs.cuda())
            target = target.cuda()
            loss = criterion(output, target)
            prec1, = accuracy(output, target, topk=(1,))
            top1.update(prec1, inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('top1:[%d, %5d] loss: %.5f acc:%.5f' % (epoch + 1, i + 1, losses.avg, top1.avg))
        print(time.time() - start)
        torch.save(model.state_dict(), 'resnet.pth')


def test(batch_size):
    model = resnet.resnet50(num_classes=2)
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load("resnet.pth"))
    train_dataloader = testdataloader.get_test_loader('data', batch_size)
    top1 = AverageMeter()
    for i, data in enumerate(train_dataloader):
        inputs, targets = data
        target = targets[0]
        inputs = inputs.clone().detach().float()
        target = target.cuda()
        output = model(inputs.cuda())
        prec1, = accuracy(output, target, topk=(1,))
        top1.update(prec1, inputs.size(0))
    print('top1:acc_1:%.5f' % (top1.avg))


if __name__ == '__main__':
    train(16)
    test(16)
