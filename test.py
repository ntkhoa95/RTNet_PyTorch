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
from utils.util import *
from models import RTFNet
import warnings

from utils.util import compute_results, visualise, SegMetrics
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Testing with PyTorch')
parser.add_argument('--dataset', type=str, default='gmrpd', help='choosing dataset for training session')
parser.add_argument('--experiment_name', type=str, default='gmrpd_manual')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes in selected dataset')
parser.add_argument('--dataroot', type=str, default='/media/asr/Data/IVAM_Lab/Master_Thesis/FuseNet/gmrpd_ds_4', help='directory of the loading data')
parser.add_argument('--resize_h', type=int, default=480, help='target resizing height')
parser.add_argument('--resize_w', type=int, default=640, help='target resizing width')

parser.add_argument('--model_name', type=str, default='RTFNet', help='chooosing model for training session')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--load_epoch', type=str, default='latest', help='choosing epoch for testing phase') # TODO: 'best_iou', 'best_precision'
parser.add_argument('--batch_size', type=int, default=4, help='number of images in a loading batch')
parser.add_argument('--gpu_ids', type=int, default=0, help='setting index of GPU for traing, "-1" for CPU')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for loading data')
parser.add_argument('--visualization_flag', type=bool, default=True, help='setting flag for visualizing results during training session')

parser.add_argument('--verbose', type=bool, default=False, help='if specified, debugging size of each part of model')

args = parser.parse_args()

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_ids)
    model = eval(args.model_name)(n_class=args.num_classes, num_resnet_layers=18, verbose=args.verbose)
    print(model)
    model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    load_network(model, args.load_epoch, model_checkpoint_dir)
    model.eval()
    
    if args.gpu_ids >= 0: model.cuda(args.gpu_ids)

    # Prepare folder
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    experiment_ckpt_dir = os.path.join(args.checkpoint_dir, args.experiment_name, args.load_epoch)
    os.makedirs(experiment_ckpt_dir, exist_ok=True)

    # Creating writter to save training logs
    writer = SummaryWriter(f"{experiment_ckpt_dir}/tensorboard_log")
    os.chmod(f"{experiment_ckpt_dir}/tensorboard_log", stat.S_IRWXO)

    test_dataset  = GMRPD_dataset(data_path=args.dataroot, phase='test', transform=False, experiment_name=args.experiment_name)

    test_loader   = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                    shuffle=False,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)                                            

    judge = SegMetrics(num_classes=args.num_classes)
    conf_mat = np.zeros((args.num_classes, args.num_classes))
    if args.dataset == 'gmrpd':
        label_list = ['Unknown', 'Drivable Area', 'Road Anomalies']
    else:
        label_list = [f'label_{i}' for i in range(args.num_classes)]
    testing_output_file = os.path.join(experiment_ckpt_dir, 'result_log.txt')
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
                visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), \
                    experiment_name=os.path.join(args.experiment_name, args.load_epoch), dataset_name='gmrpd', phase='test')
    
    # Compute confusion matrix
    pre, rec, iou = compute_results(conf_mat)
    writer.add_scalar('Test/Average_Precision', pre.mean())
    writer.add_scalar('Test/Average_Recall', rec.mean())
    writer.add_scalar('Test/Average_IoU', iou.mean())
    for i in range(len(pre)):
        writer.add_scalar('Test(class)/Precision_Class_%s' % label_list[i], pre[i])
        writer.add_scalar('Test(class)/Recall_Class_%s' % label_list[i], rec[i])
        writer.add_scalar('Test(class)/IoU_Class_%s' % label_list[i], iou[i])
    
    with open(testing_output_file, 'w') as file:
        file.write(f"\n # Tesing Phase \n")
        for i in range(args.num_classes):
            file.write("%s : Precision: %0.4f, Recall: %0.4f, IoU: %0.4f \n" % (label_list[i], 100*pre[i], 100*rec[i], 100*iou[i]))
        file.write("Mean Precision: %0.4f, Mean Recall: %0.4f, Mean IoU: %0.4f \n" % (100*np.nanmean(pre), 100*np.nanmean(rec), 100*np.nanmean(iou)))
        file.write("-" * 70)

    # acc, acc_results = judge.pixel_acc()
    precision = judge.precision_per_class()
    recall = judge.recall_per_class()
    miou = judge.miou_per_class()