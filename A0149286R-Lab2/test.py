import torch
from model import MyModel
from dataset import MyDataset
from train import AverageMeter, topk_accuracy

def test():
    my_model = MyModel()
    my_model = nn.DataParallel(my_model.cuda().float())
    my_model.eval()
    batch_size = 64

    pretrained_dict = torch.load('ckpt_e9.pth')
    my_model.load_state_dict(pretrained_dict)
    test_set = MyDataset(is_train=False)


    # mp.set_start_method('spawn', force=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, \
                      shuffle=False)

    # test accuracy
    test_acc = AverageMeter()
    for i, (image, intention, label) in enumerate(test_loader):
        image, intention, label = image.cuda(), intention.cuda(), label.cuda().view(-1)
        with torch.no_grad():
            prediction = my_model(image, intention)
            acc = topk_accuracy(2, prediction, label)
            test_acc.update(acc)

        if i % 10 == 9:
            print(f'test: iteration {i} / {len(test_loader)}, '
                  f'test accuracy {test_acc.avg:.4f}')

    print(f'evaluation finished, val acc {test_acc.avg:.4f}')

if __name__ == "__main__":
    test()