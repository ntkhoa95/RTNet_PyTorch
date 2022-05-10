import os, torch
import numpy as np
import stat, argparse
from tqdm import tqdm

import torch.nn.functional as F
from torch.autograd import Variable
from datasets.GMRPD_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from models import RTFNet

import warnings

from utils.util import compute_results, visualise, SegMetrics
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Training with PyTorch')
parser.add_argument('--dataset', type=str, default='gmrpd', help='choosing dataset for training session')
parser.add_argument('--experiment_name', type=str, default='gmrpd_manual')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes in selected dataset')
parser.add_argument('--using_class_weights', type=bool, default=True, help='flag for using class weights for training')
parser.add_argument('--dataroot', type=str, default='/media/asr/Data/IVAM_Lab/Master_Thesis/FuseNet/gmrpd_ds_4', help='directory of the loading data')
parser.add_argument('--resize_h', type=int, default=480, help='target resizing height')
parser.add_argument('--resize_w', type=int, default=640, help='target resizing width')

parser.add_argument('--model_name', type=str, default='RTFNet', help='chooosing model for training session')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs for training session')
parser.add_argument('--batch_size', type=int, default=4, help='number of images in a loading batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--gpu_ids', type=int, default=0, help='setting index of GPU for traing, "-1" for CPU')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for loading data')
parser.add_argument('--lr_decay', type=float, default=0.95, help='weight decay for adjusting learning rate')
parser.add_argument('--augmentation', type=bool, default=True, help='setting random augmentation')
parser.add_argument('--save_every', type=int, default=50, help='save model every defined epochs')
parser.add_argument('--visualization_flag', type=bool, default=True, help='setting flag for visualizing results during training session')

parser.add_argument('--verbose', type=bool, default=False, help='if specified, debugging size of each part of model')

args = parser.parse_args()

def train(epoch, model, train_loader, optimizer):
    model.train()
    for it, (imgs, labels, names) in tqdm(enumerate(train_loader)):
        imgs = Variable(imgs).cuda(args.gpu_ids)
        labels = Variable(labels).cuda(args.gpu_ids)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        optimizer.step()

        if args.visualization_flag:
            visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), \
                        experiment_name=args.experiment_name, dataset_name='gmrpd', phase='train')

        if acc_iter['train'] % 1 == 0:
            writer.add_scalar("Train/loss", loss, acc_iter['train'])
        acc_iter['train'] += 1

def validation(epoch, model, val_loader):
    model.eval()
    with torch.no_grad():
        for it, (imgs, labels, names) in tqdm(enumerate(val_loader)):
            imgs = Variable(imgs).cuda(args.gpu_ids)
            labels = Variable(labels).cuda(args.gpu_ids)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            if args.visualization_flag:
                visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), experiment_name=args.experiment_name, dataset_name='gmrpd', phase='val')
            if acc_iter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, acc_iter['val'])
            acc_iter['val'] += 1

def testing(epoch, model, test_loader):
    model.eval()
    judge = SegMetrics(num_classes=args.num_classes)
    conf_mat = np.zeros((args.num_classes, args.num_classes))
    if args.dataset == 'gmrpd':
        label_list = ['Unknown', 'Drivable Area', 'Road Anomalies']
    else:
        label_list = [f'label_{i}' for i in range(args.num_classes)]
    testing_output_file = os.path.join(experiment_ckpt_dir, 'testing_output.txt')
    with torch.no_grad():
        for it, (imgs, labels, names) in tqdm(enumerate(test_loader)):
            imgs = Variable(imgs).cuda(args.gpu_ids)
            labels = Variable(labels).cuda(args.gpu_ids)
            logits = model(imgs)
            labels_numpy = labels.cpu().numpy().squeeze().flatten()
            preds  = logits.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=labels_numpy, y_pred=preds, labels=list(range(args.num_classes)))
            conf_mat += conf
            judge.add_batch(preds, labels_numpy)
            if args.visualization_flag:
                visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), experiment_name=args.experiment_name, dataset_name='gmrpd', phase='test')
    
    # Compute confusion matrix
    pre, rec, iou = compute_results(conf_mat)
    writer.add_scalar('Test/Average_Precision', pre.mean(), epoch)
    writer.add_scalar('Test/Average_Recall', rec.mean(), epoch)
    writer.add_scalar('Test/Average_IoU', iou.mean(), epoch)
    for i in range(len(pre)):
        writer.add_scalar('Test(class)/Precision_Class_%s' % label_list[i], pre[i], epoch)
        writer.add_scalar('Test(class)/Recall_Class_%s' % label_list[i], rec[i], epoch)
        writer.add_scalar('Test(class)/IoU_Class_%s' % label_list[i], iou[i], epoch)
    
    if epoch == 0:
        with open(testing_output_file, 'w') as file:
            file.write("Number Of Training: %s, Initial Learning Rate: %s, Batch Size: %s \n" % (args.num_epochs, args.learning_rate, args.batch_size))
            file.write("Number of Classes: %s" %args.num_classes)
    
    with open(testing_output_file, 'a') as file:
        file.write(f"\n # Tesing Epoch Num: {epoch} \n")
        for i in range(args.num_classes):
            file.write("%s : Precision: %0.4f, Recall: %0.4f, IoU: %0.4f \n" % (label_list[i], 100*pre[i], 100*rec[i], 100*iou[i]))
        file.write("Mean Precision: %0.4f, Mean Recall: %0.4f, Mean IoU: %0.4f \n" % (100*np.nanmean(pre), 100*np.nanmean(rec), 100*np.nanmean(iou)))
        file.write("-" * 70)

    # acc, acc_results = judge.pixel_acc()
    precision = judge.precision_per_class()
    recall = judge.recall_per_class()
    miou = judge.miou_per_class()
    return precision, recall, miou

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_ids)
    
    model = eval(args.model_name)(n_class=args.num_classes, num_resnet_layers=18, verbose=args.verbose)
    print(model)
    if args.gpu_ids >= 0: model.cuda(args.gpu_ids)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    if args.using_class_weights:
        if args.dataset == 'gmrpd':
            if args.experiment_name.endswith('manual'):
                class_weights = torch.from_numpy(np.loadtxt(os.path.join(args.dataroot, "class_weights_manual"), delimiter=',').astype(np.float32))
            elif args.experiment_name.endswith('sslg'):
                class_weights = torch.from_numpy(np.loadtxt(os.path.join(args.dataroot, "class_weights_sslg"), delimiter=',').astype(np.float32))
            elif args.experiment_name.endswith('ALSDL'):
                class_weights = torch.from_numpy(np.loadtxt(os.path.join(args.dataroot, "class_weights_ALSDL"), delimiter=',').astype(np.float32))
            elif args.experiment_name.endswith('agsl'):
                class_weights = torch.from_numpy(np.loadtxt(os.path.join(args.dataroot, "class_weights_agsl"), delimiter=',').astype(np.float32))

    if args.gpu_ids >= 0: class_weights=class_weights.cuda(args.gpu_ids)

    # Prepare folder
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    experiment_ckpt_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(experiment_ckpt_dir, exist_ok=True)
    os.chmod(experiment_ckpt_dir, stat.S_IRWXO)

    # Creating writter to save training logs
    writer = SummaryWriter(f"{experiment_ckpt_dir}/tensorboard_log")
    os.chmod(f"{experiment_ckpt_dir}/tensorboard_log", stat.S_IRWXO)

    print("Experiment name: ", args.experiment_name)
    print("Training on GPU: ", args.gpu_ids)
    print("Training log saved to: ", experiment_ckpt_dir)

    # Setting datasets
    train_dataset = GMRPD_dataset(data_path=args.dataroot, phase='train', transform=True, experiment_name=args.experiment_name)
    val_dataset   = GMRPD_dataset(data_path=args.dataroot, phase='val', transform=False, experiment_name=args.experiment_name)
    test_dataset  = GMRPD_dataset(data_path=args.dataroot, phase='test', transform=False, experiment_name=args.experiment_name)

    train_loader  = DataLoader(dataset=train_dataset, \
                                batch_size=args.batch_size,
                                    shuffle=True,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)
    val_loader    = DataLoader(dataset=val_dataset, \
                                batch_size=args.batch_size,
                                    shuffle=False,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)
    test_loader   = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                    shuffle=False,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)                                            

    best_precision = 0
    best_miou = 0
    acc_iter = {"train": 0, "val": 0}
    for epoch in range(1, args.num_epochs+1):
        print(f"\nTraining {args.model_name} | Epoch {epoch}/{args.num_epochs}")
        train(epoch, model, train_loader, optimizer)
        validation(epoch, model, val_loader)
        checkpoint_model_file = os.path.join(experiment_ckpt_dir, 'latest_model.pth')
        print('Saving latest checkpoint model!')
        torch.save(model.state_dict(), checkpoint_model_file)

        if epoch % args.save_every == 0:
            checkpoint_model_file = os.path.join(experiment_ckpt_dir, str(epoch)+'_model.pth')
            print('Saving checkpoint model!')
            torch.save(model.state_dict(), checkpoint_model_file)

        precision, recall, miou = testing(epoch, model, test_loader)

        if epoch == 1:
            best_precision = precision
            best_miou = miou
            checkpoint_pre_model_file = os.path.join(experiment_ckpt_dir, 'best_precision_model.pth')
            torch.save(model.state_dict(), checkpoint_pre_model_file)
            checkpoint_miou_model_file = os.path.join(experiment_ckpt_dir, 'best_iou_model.pth')
            torch.save(model.state_dict(), checkpoint_miou_model_file)
        else:
            if precision > best_precision:
                checkpoint_pre_model_file = os.path.join(experiment_ckpt_dir, 'best_precision_model.pth')
                torch.save(model.state_dict(), checkpoint_pre_model_file)
            if miou > best_miou:
                checkpoint_miou_model_file = os.path.join(experiment_ckpt_dir, 'best_iou_model.pth')
                torch.save(model.state_dict(), checkpoint_miou_model_file)
        scheduler.step()

# python train.py --experiment_name gmrpd_manual
# python train.py --experiment_name gmrpd_ALSDL
# python train.py --experiment_name gmrpd_agsl
# python train.py --experiment_name gmrpd_sslg