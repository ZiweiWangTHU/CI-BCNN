from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import argparse
import torch.nn as nn
from tqdm import tqdm

from models import resnet
from RL_network import PGAgent
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util import adjust_optimizer


def discretization(alpha):
    alpha_prime = 0
    if alpha < -0.04:
        alpha_prime = -0.04
    # if alpha > -0.006:
    #     alpha_prime = -0.006
    # if alpha > -0.004:
    #     alpha_prime = -0.004
    if alpha > 0:
        alpha_prime = 0
    # if alpha > 0.004:
    #     alpha_prime = 0.004
    # if alpha > 0.006:
    #     alpha_prime = 0.006
    if 0.04 < alpha:
        alpha_prime = 0.04
    # if alpha > 0.004:
    #     alpha_prime = 0.004
    return alpha_prime


# rl training part
def dependency_mining(dim_featuremaps):  # rl
    """
        dim_featuremaps: the list of features dimension of each layer
        dependency_state: whether a feature is dependent of another feature
        influence_state: how much the influence is if the dependence exists
        xx: teacher feature
        yy: student feature
    """

    # initialize 
    global rl, xx, yy, dependency_state, influence_state
    model.set(xx, yy, influence_state)
    max_connection = 100

    next_influence_state_dis = []
    for i in range(len(dim_featuremaps)):
        next_influence_state_dis.append(np.zeros((max_dim, max_dim)))

    # dependence mining
    model.eval()
    last_dependency_state, last_influence_state = dependency_state, influence_state

    with torch.no_grad():
        for data, target in tqdm(trainloader, ascii=True):
            data, target = data.cuda(), target.cuda()
            loss_bf = criterion(model(data), target)

            input = []
            for i in range(len(dim_featuremaps)):
                input.append(np.vstack(([dependency_state[i]], [influence_state[i]])))
            input = np.vstack([[input]])

            action1, action2, next_dependency_state_con, next_influence_state_con = rl.act(input)
            reward1, reward2, reward3 = [], [], []

            flag = 0  # whether a connection has been found
            fflag = 0  # whether a connection is removed

            for i in range(len(dim_featuremaps)):
                # add a connection
                if action1[i] > -1:
                    x_, y_ = action1[i] // dim_featuremaps[i], action1[i] % dim_featuremaps[i]
                    if dependency_state[i][x_, y_] == 1 or x_ == y_:
                        flag = 1
                    if flag == 0 or (len(xx[i]) == 0 and x_ != y_):
                        xx[i].append(x_)
                        yy[i].append(y_)

                # discretize the influence matrix
                next_influence_state_dis[i] = np.zeros((max_dim, max_dim))
                for j in range(len(xx[i])):
                    next_influence_state_dis[i][xx[i][j]][yy[i][j]] = discretization(
                        next_influence_state_con[i][xx[i][j]][yy[i][j]])

                # judge the effect
                model.set(xx, yy, next_influence_state_dis)
                loss_af1 = criterion(model(data), target)
                diff = (loss_af1 - loss_bf).data.cpu().numpy()
                features = model.feature
                inform_diff = 0
                N = 1
                for l in range(len(features)):
                    for t in range(len(xx[l])):
                        abs_diff = torch.abs(features[l][:, xx[l][t], :, :]) - torch.abs(features[l][:, yy[l][t], :, :])
                        inform_diff += torch.sum(torch.sign(abs_diff)).item()
                        N += abs_diff.size(0) * abs_diff.size(1) * abs_diff.size(2)
                if diff < -0.003:
                    reward1.append(1 + inform_diff / N)
                elif diff > 0.003:
                    reward1.append(-1 + inform_diff / N)
                else:
                    reward1.append(0 + inform_diff / N)
                if reward1[i] <= 0:
                    for j in range(len(xx[i])):
                        if xx[i][j] == action1[i] // dim_featuremaps[i] and yy[i][j] == action1[i] % dim_featuremaps[i]:
                            del xx[i][j]
                            del yy[i][j]
                            break
                # print((loss_af1 - loss_bf).data.cpu().numpy())

                # remove a connection
                temp_x = 0
                temp_y = 0
                if action2[i] > -1:
                    x_, y_ = action2[i] // dim_featuremaps[i], action2[i] % dim_featuremaps[i]
                    if dependency_state[i][x_, y_] == 1:
                        fflag = 1
                        for j in range(len(xx[i])):
                            if xx[i][j] == x_ and yy[i][j] == y_:
                                temp_x = xx[i][j]
                                temp_y = yy[i][j]
                                del xx[i][j]
                                del yy[i][j]
                                break

                # discretize the influence matrix
                next_influence_state_dis[i] = np.zeros((max_dim, max_dim))
                for j in range(len(xx[i])):
                    next_influence_state_dis[i][xx[i][j]][yy[i][j]] = discretization(
                        next_influence_state_con[i][xx[i][j]][yy[i][j]])

                # judge the effect
                model.set(xx, yy, next_influence_state_dis)
                loss_af2 = criterion(model(data), target)
                diff = (loss_af2 - loss_af1).data.cpu().numpy()
                features = model.feature
                inform_diff = 0
                N = 1
                for l in range(len(features)):
                    for t in range(len(xx[l])):
                        abs_diff = torch.abs(features[l][:, xx[l][t], :, :]) - torch.abs(features[l][:, yy[l][t], :, :])
                        inform_diff += torch.sum(torch.sign(abs_diff)).item()
                        N += abs_diff.size(0) * abs_diff.size(1) * abs_diff.size(2)
                if diff < -0.003 and reward1[i] > 0:
                    reward2.append(1 + inform_diff / N)
                elif diff > 0.003:
                    reward2.append(-1 + inform_diff / N)
                else:
                    reward2.append(0 + inform_diff / N)
                if reward2[i] <= 0 and dependency_state[i][
                    action2[i] // dim_featuremaps[i], action2[i] % dim_featuremaps[i]] == 1:
                    xx[i].append(temp_x)
                    yy[i].append(temp_y)

                # delete the most ambiguious connection if the connection is beyond max
                min_ambiguity = float('inf')
                min_index = -1
                if len(xx[i]) > max_connection and flag == 0:
                    for j in range(len(xx[i])):
                        temp_ambiguity = influence_state[i][xx[i][j]][yy[i][j]]
                        if min_ambiguity > temp_ambiguity:
                            min_ambiguity = temp_ambiguity
                            min_index = j

                    action2[i] = xx[i][min_index] * dim_featuremaps[i] + yy[i][min_index]
                    del xx[i][min_index]
                    del yy[i][min_index]

                # judge the effect
                next_influence_state_dis[i] = np.zeros((max_dim, max_dim))
                for j in range(len(xx[i])):
                    next_influence_state_dis[i][xx[i][j]][yy[i][j]] = discretization(
                        next_influence_state_con[i][xx[i][j]][yy[i][j]])
                model.set(xx, yy, next_influence_state_dis)
                loss_af2 = criterion(model(data), target)
                diff = (loss_af2 - loss_af1).data.cpu().numpy()
                features = model.feature
                inform_diff = 0
                N = 1
                for l in range(len(features)):
                    for t in range(len(xx[l])):
                        abs_diff = torch.abs(features[l][:, xx[l][t], :, :]) - torch.abs(features[l][:, yy[l][t], :, :])
                        inform_diff += torch.sum(torch.sign(abs_diff)).item()
                        N += abs_diff.size(0) * abs_diff.size(1) * abs_diff.size(2)
                if diff < -0.003:
                    reward2[i] = 1 + inform_diff / N
                elif diff > 0.003:
                    reward2[i] = -1 + inform_diff / N
                else:
                    reward2[i] = 0 + inform_diff / N

                # judge the overall effect
                if (loss_af2 - loss_bf).data.cpu().numpy() < -0.003:
                    reward3.append(1)
                elif (loss_af2 - loss_bf).data.cpu().numpy() > 0.003:
                    reward3.append(-1)
                else:
                    reward3.append(0)

                ## construct new dependency_state
                dependency_state[i] = np.zeros((max_dim, max_dim))
                influence_state[i] = np.zeros((max_dim, max_dim))
                if len(xx[i]) > 0:
                    for j in range(len(xx[i])):
                        dependency_state[i][xx[i][j], yy[i][j]] = 1
                        influence_state[i][xx[i][j], yy[i][j]] = discretization(
                            next_influence_state_con[i][xx[i][j], yy[i][j]])
                if np.sum(dependency_state[i]) == 0:
                    dependency_state[i] = np.random.rand(max_dim, max_dim)
                    influence_state[i] = np.random.rand(max_dim, max_dim)

            # all layers have been mined
            # then train together
            rl.remember(last_dependency_state, last_influence_state,
                        next_dependency_state_con, next_influence_state_con, next_influence_state_dis,
                        action1, action2,
                        50 * (np.array(reward1) - 0.03).tolist(), 50 * (np.array(reward2) - 0.03).tolist(),
                        50 * (np.array(reward3) - 0.03).tolist())
            rl.train()
    print(xx)
    print(yy)
    print(reward1)
    return xx, yy, next_influence_state_dis


def save_state(model, best_acc, xx, yy, influence_state):
    print('==> Saving model ...')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, './results/resnet_alpha.pth.tar')

    if xx:
        np.save("./results/xx.npy", xx)
        np.save("./results/yy.npy", yy)
        np.save("./results/influence_state.npy", influence_state)


def train(epoch):
    # switch to train mode1
    model.train()
    for i, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(inputs), len(trainloader.dataset),
                100. * i / len(trainloader), loss.item()))
    return


def test(xx=None, yy=None, influence_state=None):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for inputs, targets in testloader:
        # measure data loading time
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        output = model(inputs)
        test_loss += criterion(output, targets).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(testloader.dataset)
    if acc > best_acc:
        best_acc = acc
        save_state(model, acc, xx, yy, influence_state)

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))


def adjust_learning_rate(optimizer, epoch):
    update_list = [150]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data',
                        help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet',
                        help='the architecture for the network: resnet')
    parser.add_argument('--lr', action='store', default='0.01', type=float,
                        help='the intial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--CI', action='store', default=None,
                        help='the path to the channel interaction files')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', default=0,  type=int,
                        help='starting epoch')

    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize, ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define the model
    if args.arch == 'resnet':
        model = resnet.resnet_binary()
    else:
        raise Exception(args.arch + ' is currently not supported')

    # initialize the model
    if not args.pretrained:
        best_acc = 0

    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        # best_acc = 0.9068
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()

    # define solver and criterion
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params': [value], 'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'key': key}]

    # optimizer = optim.Adam(params, lr=0.001, weight_decay=0.00001)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    regime = getattr(model, 'regime', {0: {'optimizer': 'SGD',
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})

    criterion = nn.CrossEntropyLoss()

    # do the evaluation if specified
    if args.evaluate:
        xx = np.load(args.CI + "/xx.npy", allow_pickle=True)
        yy = np.load(args.CI + "/yy.npy", allow_pickle=True)
        influence_state = np.load(args.CI + "/influence_state.npy", allow_pickle=True)
        # model.set(xx, yy, influence_state)
        test(xx, yy, influence_state)
        exit(0)

    # the rl settings
    dim_featuremaps = [80, 80, 80, 80, 160, 160, 160, 160, 160, 320, 320, 320, 320, 320]
    rl = PGAgent(dim_featuremaps)
    max_dim = max(dim_featuremaps)

    # initializing
    xx, yy, dependency_state, influence_state = [], [], [], []
    for i in range(len(dim_featuremaps)):
        dependency_state.append(np.random.rand(max_dim, max_dim) - 0.5)
        influence_state.append(np.random.rand(max_dim, max_dim) - 0.5)
        xx.append([])
        yy.append([])

    mining = best_acc > 89.  # start mining
    for epoch in range(args.start_epoch, 3000):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        if not mining:
            train(epoch)
            test()
            if best_acc > 89.:
                mining = True
                print("Starting Mining...")
            continue

        if epoch % 500 < 50:
            train(epoch)
            test(xx, yy, influence_state)
        else:
            xx, yy, influence_state = dependency_mining(dim_featuremaps)
            test(xx, yy, influence_state)
