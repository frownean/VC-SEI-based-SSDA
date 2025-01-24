from get_data import Source_Dataset, SemiSupervised_Target_Dataset
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from fe import *
from cls import *
import random
import os
from fun import GradientReversal
from Island_Loss import IslandLoss
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    def __init__(
        self,
        source_batch_size: int = 32,
        supervised_target_batch_size: int = 32,
        unsupervised_target_batch_size: int = 128,
        test_batch_size: int = 32,
        epochs: int = 2,
        lr: float = 0.001,
        lr_cent: float = 0.05,
        source_ft: int = 2,
        target_ft: int = 62,
        pre_fe_path: str = r'model_weight/fe.pth',
        pre_cls_path: str = r'model_weight/cls.pth',
        device_num: int = 0,
        rand_num: int = 30,
        n_classes: int = 16
    ):
        self.source_batch_size = source_batch_size
        self.supervised_target_batch_size = supervised_target_batch_size
        self.unsupervised_target_batch_size = unsupervised_target_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_cent = lr_cent
        self.source_ft = source_ft
        self.target_ft = target_ft
        self.pre_fe_path = pre_fe_path
        self.pre_cls_path = pre_cls_path
        self.device_num = device_num
        self.rand_num = rand_num
        self.n_classes = n_classes


conf = Config()


def train(feature_extractor, feature_classifier, discriminator, source_train_dataloader, supervised_target_train_dataloader, unsupervised_target_train_dataloader, loss_island, optimizer, optimizer_cent, optimizer_disc, epoch):
    feature_extractor.train()
    feature_classifier.train()
    discriminator.train()
    n_batches = min(len(source_train_dataloader), len(supervised_target_train_dataloader), len(unsupervised_target_train_dataloader))
    total_source_label_loss = total_target_label_loss = 0
    total_domain_loss = total_source_label_accuracy = total_target_label_accuracy = 0
    total_island_loss = 0
    for (source_x, source_labels), (supervised_target_x, supervised_target_labels), (unsupervised_target_x, _) in zip(source_train_dataloader, supervised_target_train_dataloader, unsupervised_target_train_dataloader):
        # data
        x = torch.cat([source_x, supervised_target_x, unsupervised_target_x])
        x = x.to(device)
        domain_y = torch.cat([torch.ones(source_x.shape[0]), torch.zeros(supervised_target_x.shape[0]), torch.zeros(unsupervised_target_x.shape[0])])
        domain_y = domain_y.to(device)
        source_labels = source_labels.long().to(device)
        supervised_target_labels = supervised_target_labels.long().to(device)

        # FE
        fe = feature_extractor(x)

        isla_fe = fe[:source_x.shape[0]+supervised_target_x.shape[0]]
        isla_labels = torch.cat((source_labels, supervised_target_labels))

        # disc loss
        domain_preds = discriminator(fe).squeeze()
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)

        # center loss
        island_loss = loss_island(isla_fe, isla_labels)

        # label loss
        label_preds = feature_classifier(fe[:source_x.shape[0]+supervised_target_x.shape[0]])
        source_label_preds = label_preds[:source_x.shape[0]]
        supervised_target_label_preds = label_preds[source_x.shape[0]:]
        source_classifier_output = F.log_softmax(source_label_preds, dim=1)
        supervised_target_classifier_output = F.log_softmax(supervised_target_label_preds, dim=1)
        source_label_loss = F.nll_loss(source_classifier_output, source_labels)
        target_label_loss = F.nll_loss(supervised_target_classifier_output, supervised_target_labels)

        # total loss
        loss = 0.01*domain_loss + source_label_loss + target_label_loss + 0.01*island_loss

        optimizer.zero_grad()
        optimizer_disc.zero_grad()
        optimizer_cent.zero_grad()
        loss.backward()

        optimizer.step()
        for param in discriminator.parameters():
                param.grad.data *= (1. / 0.01)
        optimizer_disc.step()
        for param in loss_island.parameters():
                param.grad.data *= (1. / 0.01)
        optimizer_cent.step()

        total_domain_loss += domain_loss.item()
        total_source_label_loss += source_label_loss.item()
        total_target_label_loss += target_label_loss.item()
        total_island_loss += island_loss.item()
        total_source_label_accuracy += (source_label_preds.max(1)[1] == source_labels).float().mean().item()
        total_target_label_accuracy += (supervised_target_label_preds.max(1)[1] == supervised_target_labels).float().mean().item()

    mean_domain_loss = total_domain_loss / n_batches
    mean_source_label_loss = total_source_label_loss / n_batches
    mean_target_label_loss = total_target_label_loss / n_batches
    mean_island_loss = total_island_loss / n_batches
    mean_source_accuracy = total_source_label_accuracy / n_batches
    mean_target_accuracy = total_target_label_accuracy / n_batches
    print('Train Epoch: {} \tdomain_Loss: {:.6f}, Source_label_loss: {:.6f}, Target_label_loss: {:.6f}, Island_loss: {:.6f}, Source_Accuracy: {:.6f}, Target_Accuracy: {:.6f} \n'.format(
        epoch,
        mean_domain_loss,
        mean_source_label_loss,
        mean_target_label_loss,
        mean_island_loss,
        mean_source_accuracy,
        mean_target_accuracy,
    ))


def val(feature_extractor, feature_classifier, discriminator, source_val_dataloader, supervised_target_val_dataloader, unsupervised_target_val_dataloader, loss_island, epoch):
    feature_extractor.eval()
    feature_classifier.eval()
    discriminator.eval()

    with torch.no_grad():
        n_batches = min(len(source_val_dataloader), len(supervised_target_val_dataloader),
                        len(unsupervised_target_val_dataloader))
        total_loss = total_source_label_loss = total_target_label_loss = 0
        total_domain_loss = total_source_label_accuracy = total_target_label_accuracy = 0
        total_island_loss = 0
        for (source_x, source_labels), (supervised_target_x, supervised_target_labels), (
        unsupervised_target_x, _) in zip(source_val_dataloader, supervised_target_val_dataloader,
                                         unsupervised_target_val_dataloader):
            # data
            x = torch.cat([source_x, supervised_target_x, unsupervised_target_x])
            x = x.to(device)
            domain_y = torch.cat([torch.ones(source_x.shape[0]), torch.zeros(supervised_target_x.shape[0]),
                                  torch.zeros(unsupervised_target_x.shape[0])])
            domain_y = domain_y.to(device)
            source_labels = source_labels.long().to(device)
            supervised_target_labels = supervised_target_labels.long().to(device)

            # FE
            x = feature_extractor(x)

            isla_fe = x[:source_x.shape[0] + supervised_target_x.shape[0]]
            isla_labels = torch.cat((source_labels, supervised_target_labels))

            # disc loss
            domain_preds = discriminator(x).squeeze()
            domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)

            # center loss
            island_loss = loss_island(isla_fe, isla_labels)

            # label loss
            label_preds = feature_classifier(x[:source_x.shape[0] + supervised_target_x.shape[0]])
            source_label_preds = label_preds[:source_x.shape[0]]
            supervised_target_label_preds = label_preds[source_x.shape[0]:]
            source_classifier_output = F.log_softmax(source_label_preds, dim=1)
            supervised_target_classifier_output = F.log_softmax(supervised_target_label_preds, dim=1)
            source_label_loss = F.nll_loss(source_classifier_output, source_labels)
            target_label_loss = F.nll_loss(supervised_target_classifier_output, supervised_target_labels)

            # total loss
            loss = 0.01*domain_loss + source_label_loss + target_label_loss + 0.01*island_loss

            total_domain_loss += domain_loss.item()
            total_source_label_loss += source_label_loss.item()
            total_target_label_loss += target_label_loss.item()
            total_island_loss += island_loss.item()
            total_loss += loss.item()
            total_source_label_accuracy += (source_label_preds.max(1)[1] == source_labels).float().mean().item()
            total_target_label_accuracy += (supervised_target_label_preds.max(1)[1] == supervised_target_labels).float().mean().item()

        mean_domain_loss = total_domain_loss / n_batches
        mean_source_label_loss = total_source_label_loss / n_batches
        mean_target_label_loss = total_target_label_loss / n_batches
        mean_island_loss = total_island_loss / n_batches
        mean_loss = total_loss / n_batches
        mean_source_accuracy = total_source_label_accuracy / n_batches
        mean_target_accuracy = total_target_label_accuracy / n_batches
        print(
            'Val Epoch: {} \tdomain_Loss: {:.6f}, Source_label_loss: {:.6f}, Target_label_loss: {:.6f}, Center_loss: {:.6f}, Source_Accuracy: {:.6f}, Target_Accuracy: {:.6f} \n'.format(
                epoch,
                mean_domain_loss,
                mean_source_label_loss,
                mean_target_label_loss,
                mean_island_loss,
                mean_source_accuracy,
                mean_target_accuracy,
            ))

    return mean_loss


def test(feature_extractor, feature_classifier, test_dataloader):
    feature_extractor.eval()
    feature_classifier.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = feature_extractor(data)
            output = feature_classifier(output)
            output = F.log_softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = 100.0 * correct / len(test_dataloader.dataset)

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            acc,
        )
    )
    return acc


def train_and_val_and_test(feature_extractor, feature_classifier,  discriminator, source_train_dataset, source_val_dataset, supervised_target_train_dataset,
                           supervised_target_val_dataset, unsupervised_target_train_dataset, unsupervised_target_val_dataset, loss_center, optimizer, optimizer_cent,
                           optimizer_disc, epochs, ratio):
    current_min_test_loss = 10000
    for epoch in range(1, epochs + 1):
        source_train_dataloader = DataLoader(source_train_dataset, batch_size=conf.source_batch_size, shuffle=True, num_workers=1)
        source_val_dataloader = DataLoader(source_val_dataset, batch_size=conf.source_batch_size, shuffle=True, num_workers=1)

        supervised_target_train_dataloder = DataLoader(supervised_target_train_dataset, batch_size=conf.supervised_target_batch_size, shuffle=True, num_workers=1)
        supervised_target_val_dataloder = DataLoader(supervised_target_val_dataset, batch_size=conf.supervised_target_batch_size, shuffle=True, num_workers=1)

        unsupervised_target_train_dataloder = DataLoader(unsupervised_target_train_dataset, batch_size=conf.unsupervised_target_batch_size, shuffle=True, num_workers=1)
        unsupervised_target_val_dataloder = DataLoader(unsupervised_target_val_dataset, batch_size=conf.unsupervised_target_batch_size, shuffle=True, num_workers=1)
        train(feature_extractor, feature_classifier, discriminator, source_train_dataloader, supervised_target_train_dataloder, unsupervised_target_train_dataloder, loss_center, optimizer,
              optimizer_cent, optimizer_disc, epoch)
        val_loss = val(feature_extractor, feature_classifier, discriminator, source_val_dataloader, supervised_target_val_dataloder, unsupervised_target_val_dataloder, loss_center, epoch)
        if val_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                    current_min_test_loss, val_loss))
            current_min_test_loss = val_loss
            torch.save(feature_extractor, r'model_weight/ssda_{}fe.pth'.format(ratio))
            torch.save(feature_classifier, r'model_weight/ssda_{}cls.pth'.format(ratio))
            torch.save(discriminator, 'model_weight/ssda_{}dom.pth'.format(ratio))
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")


def main(ratio):

    RANDOM_SEED = 300
    set_seed(RANDOM_SEED)

    feature_extractor = torch.load(conf.pre_fe_path).to(device)

    clf = torch.load(conf.pre_cls_path).to(device)

    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Linear(256, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

# source_dataset
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Source_Dataset(conf.source_ft, conf.rand_num)
    print(Counter(Y_train))
    source_train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    source_val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    source_test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    source_test_loader = DataLoader(source_test_dataset, batch_size=conf.source_batch_size, shuffle=True, num_workers=1)

# target_dataset
    X_supervised_train, X_unsupervised_train, X_supervised_val, X_unsupervised_val, X_test, Y_supervised_train, Y_unsupervised_train, Y_supervised_val, Y_unsupervised_val, Y_test = SemiSupervised_Target_Dataset(conf.target_ft, conf.rand_num, ratio)
    #supervised
    supervised_target_train_dataset = TensorDataset(torch.Tensor(X_supervised_train), torch.Tensor(Y_supervised_train))
    supervised_target_val_dataset = TensorDataset(torch.Tensor(X_supervised_val), torch.Tensor(Y_supervised_val))
    #unsupervised
    unsupervised_target_train_dataset = TensorDataset(torch.Tensor(X_unsupervised_train), torch.Tensor(Y_unsupervised_train))
    unsupervised_target_val_dataset = TensorDataset(torch.Tensor(X_unsupervised_val), torch.Tensor(Y_unsupervised_val))
    #test
    target_test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    target_test_loader = DataLoader(target_test_dataset, batch_size=conf.test_batch_size, shuffle=True, num_workers=1)

    loss_island = IslandLoss(features_dim=1152, num_class=16, lamda=1., lamda1=10., scale=1.0, batch_size=18)

    optim = torch.optim.Adam(list(feature_extractor.parameters()) + list(clf.parameters()), lr=conf.lr, weight_decay=0)
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=conf.lr, weight_decay=0)
    optim_centloss = torch.optim.Adam(loss_island.parameters(), lr=conf.lr_cent, weight_decay=0)

    train_and_val_and_test(feature_extractor=feature_extractor,
                           feature_classifier=clf,
                           discriminator=discriminator,
                           source_train_dataset=source_train_dataset,
                           source_val_dataset=source_val_dataset,
                           supervised_target_train_dataset=supervised_target_train_dataset,
                           supervised_target_val_dataset=supervised_target_val_dataset,
                           unsupervised_target_train_dataset=unsupervised_target_train_dataset,
                           unsupervised_target_val_dataset=unsupervised_target_val_dataset,
                           loss_center=loss_island,
                           optimizer=optim,
                           optimizer_cent=optim_centloss,
                           optimizer_disc=optim_disc,
                           epochs=conf.epochs,
                           ratio=ratio
                           )

    ta_fe_path = r'model_weight/ssda_{}fe.pth'.format(ratio)
    ta_cls_path = r'model_weight/ssda_{}cls.pth'.format(ratio)
    feature_extractor = torch.load(ta_fe_path)
    clf = torch.load(ta_cls_path)
    acc1 = test(feature_extractor=feature_extractor, feature_classifier=clf, test_dataloader=target_test_loader)
    acc2 = test(feature_extractor=feature_extractor, feature_classifier=clf, test_dataloader=source_test_loader)
    f1 = open(r'./results/ssda_targetAcc.txt', 'a+')
    f1.write(str(acc1) + " " + str(ratio) + '\n')
    f2 = open(r'./results/ssda_sourceAcc.txt', 'a+')
    f2.write(str(acc2) + " " + str(ratio) + '\n')


if __name__ == '__main__':
    for ratio in [0.005, 0.01, 0.025, 0.05, 0.1]:
        main(ratio)


