import torch
import torchvision
import torch.nn as nn
import lenet
import torch.optim as optim
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np


def main():
    # 图像预处理
  transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  # batch_size = 4

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                          shuffle=False, num_workers=0)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                        shuffle=False, num_workers=0)
  
  test_data_iter = iter(testloader)
  test_image, test_label = next(test_data_iter)

  classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # def imshow(img):
  #   img = img / 2 + 0.5     # unnormalize
  #   npimg = img.numpy()
  #   plt.imshow(np.transpose(npimg, (1, 2, 0)))
  #   plt.show()

  # # print labels
  # print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(batch_size)))
  # # show images
  # imshow(torchvision.utils.make_grid(test_image))

  net = lenet.LeNet()
  loss_function = nn.CrossEntropyLoss()
  # 优化器
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  # epoch=5
  for epoch in range(5):  
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # 每次反向传播前清空梯度累积
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化器
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
          with torch.no_grad():
            outputs = net(test_image)  # [batch, 10]
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
            running_loss = 0.0

  print('Finished Training')

  save_path = './LeNet/LeNet.pth'
  torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()