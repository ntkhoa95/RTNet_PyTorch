import numpy as np
from PIL import Image
import os, torch
from torchvision.utils import save_image, make_grid

def get_palette(dataset="gmrpd"):
    """Visualizing segmentation results in colormap"""
    assert dataset in ["gmrpd", "cityscapes", "thermal"]
    if dataset == "gmrpd":
        unknown_area   = [0, 0, 255]
        drivable_area  = [0, 255, 0]
        road_anomalies = [255, 0, 0]
        palette = np.array([unknown_area, drivable_area, road_anomalies]).tolist()
    elif dataset == "thermal":
        unlabelled     = [0,     0,   0]
        car            = [64,    0, 128]
        person         = [64,   64,   0]
        bike           = [0,   128, 192]
        curve          = [0,     0, 192]
        car_stop       = [128, 128,   0]
        guardrail      = [64,   64, 128]
        color_cone     = [192, 128, 128]
        bump           = [192,  64,   0]
        palette        = np.array([unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump]).tolist()
    elif dataset == "cityscapes":
        road           = [128,  64, 128]
        sidewalk       = [244,  35, 232]
        building       = [70,   70,  70]
        wall           = [102, 102, 156]
        fence          = [190, 153, 153]
        pole           = [153, 153, 153]
        traffic_light  = [250, 170,  30]
        traffic_sigh   = [220, 220,   0]
        vegetation     = [107, 142,  35]
        terrain        = [152, 251, 152]
        sky            = [70,  130, 180]
        person         = [220,  20,  60]
        rider          = [255,   0,   0]
        car            = [0,     0, 142]
        truck          = [0,     0,  70]
        bus            = [0,    60, 100]
        train          = [0,    80, 100]
        motorcycle     = [0,     0, 230]
        bicycle        = [119,  11,  32]
        palette        = np.array([road, sidewalk, building, wall, fence, \
                                    pole, traffic_light, traffic_sigh, vegetation, \
                                        terrain, sky, person, rider, car, truck,\
                                            bus, train, motorcycle, bicycle]).tolist()
    return palette

def visualise(image_names, imgs, labels, predictions, experiment_name, dataset_name="gmrpd", phase='train'):
    # print(imgs.shape, labels.shape, predictions.shape)
    palette = get_palette(dataset=dataset_name)
    os.makedirs(f'./checkpoints/{experiment_name}/visualization/{phase}', exist_ok=True)
    if phase == 'training':
        img_name = image_names[-1].split(".")[0]

        input_rgb   = imgs[-1].cpu().numpy()[:3, :, :].transpose(1, 2, 0) * 255
        input_depth = imgs[-1].cpu().numpy()[3, :, :] * 255

        input_rgb = Image.fromarray(np.uint8(input_rgb))
        input_rgb.save((f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_rgb.png'))
        input_depth = Image.fromarray(np.uint8(input_depth))
        input_depth.save((f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_depth.png'))

        input_label = labels[-1].cpu().numpy()
        label_img   = np.zeros((input_label.shape[0], input_label.shape[1], 3), dtype=np.uint8)
        for cid in range(len(palette)):
            label_img[input_label == cid] = palette[cid]
        label_img = Image.fromarray(np.uint8(label_img))
        label_img.save(f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_label.png')

        pred = predictions[-1].cpu().numpy()
        pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(len(palette)):
            pred_img[pred == cid] = palette[cid]
        pred_img = Image.fromarray(np.uint8(pred_img))
        pred_img.save(f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_pred.png')
    else:
        for (idx, pred) in enumerate(predictions):
            img_name = image_names[idx].split(".")[0]

            input_rgb   = imgs[idx].cpu().numpy()[:3, :, :].transpose(1, 2, 0) * 255
            input_depth = imgs[idx].cpu().numpy()[3, :, :] * 255

            input_rgb = Image.fromarray(np.uint8(input_rgb))
            input_rgb.save((f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_rgb.png'))
            input_depth = Image.fromarray(np.uint8(input_depth))
            input_depth.save((f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_depth.png'))

            input_label = labels[idx].cpu().numpy()
            label_img   = np.zeros((input_label.shape[0], input_label.shape[1], 3), dtype=np.uint8)
            for cid in range(len(palette)):
                label_img[input_label == cid] = palette[cid]
            label_img = Image.fromarray(np.uint8(label_img))
            label_img.save(f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_label.png')

            pred = predictions[idx].cpu().numpy()
            pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for cid in range(len(palette)):
                pred_img[pred == cid] = palette[cid]
            pred_img = Image.fromarray(np.uint8(pred_img))
            pred_img.save(f'./checkpoints/{experiment_name}/visualization/{phase}/{img_name}_pred.png')

def compute_results(conf_total, ignore_void=False):
    n_class = conf_total.shape[0]
    if ignore_void:
        start_index = 1
    else:
        start_index = 0

    precision_per_class = np.zeros(n_class)
    recall_per_class    = np.zeros(n_class)
    iou_per_class       = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
        
class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)  # shape:(num_class, num_class)

    def pixel_acc(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum() 
        return round(acc * 100, 2)

    def pixel_acc_per_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print_acc_per_class(acc)
        acc = np.nanmean(acc)
        return round(acc * 100, 2)

    def miou_per_class(self):
        miou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) +
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix)
        )
        # print mIoU of each class
        print_iou_per_class(miou)
        miou = np.nanmean(miou)
        return round(miou * 100, 2)

    def fwiou_per_class(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) +
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix)
        )
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou
    
    def precision_per_class(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        print_precision_per_class(precision)
        precision = np.nanmean(precision)
        return precision

    def recall_per_class(self):
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        print_recall_per_class(recall)
        recall = np.nanmean(recall)
        return recall

    def _generate_matrix(self, pred, tgt):
        mask = (tgt >= 0) & (tgt < self.num_classes)
        label = self.num_classes * tgt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, pred, tgt):
        assert pred.shape == tgt.shape
        self.confusion_matrix += self._generate_matrix(pred, tgt)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)



def print_acc_per_class(acc):
    if len(acc) == 3:
        print('-----------Accuracy Per Class-----------')
        print('unknown      : {:.6f}'.format(acc[0] * 100.), '%\t')
        print('road         : {:.6f}'.format(acc[1] * 100.), '%\t')
        print('obstacle     : {:.6f}'.format(acc[2] * 100.), '%\t')
        print('Mean Accuracy: {:.6f}'.format((acc[0] + acc[1] + acc[2]) / 3 * 100.))

    elif len(acc) == 12:
        print('-----------Accuracy Per Class-----------')
        print('sky         : {:.6f}'.format(acc[0] * 100.), '%\t')
        print('building    : {:.6f}'.format(acc[1] * 100.), '%\t')
        print('pole        : {:.6f}'.format(acc[2] * 100.), '%\t')
        print('road        : {:.6f}'.format(acc[3] * 100.), '%\t')
        print('sidewalk    : {:.6f}'.format(acc[4] * 100.), '%\t')
        print('tree        : {:.6f}'.format(acc[5] * 100.), '%\t')
        print('sign symbol : {:.6f}'.format(acc[6] * 100.), '%\t')
        print('fence       : {:.6f}'.format(acc[7] * 100.), '%\t')
        print('car         : {:.6f}'.format(acc[8] * 100.), '%\t')
        print('pedestrian  : {:.6f}'.format(acc[9] * 100.), '%\t')
        print('bicyclist   : {:.6f}'.format(acc[10] * 100.), '%\t')
        print('void        : {:.6f}'.format(acc[11] * 100.), '%\t')
    else:
        print('-----------Accuracy Per Class-----------')
        print('road         : {:.6f}'.format(acc[0] * 100.), '%\t')
        print('sidewalk     : {:.6f}'.format(acc[1] * 100.), '%\t')
        print('building     : {:.6f}'.format(acc[2] * 100.), '%\t')
        print('wall         : {:.6f}'.format(acc[3] * 100.), '%\t')
        print('fence        : {:.6f}'.format(acc[4] * 100.), '%\t')
        print('pole         : {:.6f}'.format(acc[5] * 100.), '%\t')
        print('traffic light: {:.6f}'.format(acc[6] * 100.), '%\t')
        print('traffic sign : {:.6f}'.format(acc[7] * 100.), '%\t')
        print('vegetation   : {:.6f}'.format(acc[8] * 100.), '%\t')
        print('terrain      : {:.6f}'.format(acc[9] * 100.), '%\t')
        print('sky          : {:.6f}'.format(acc[10] * 100.), '%\t')
        print('person       : {:.6f}'.format(acc[11] * 100.), '%\t')
        print('rider        : {:.6f}'.format(acc[12] * 100.), '%\t')
        print('car          : {:.6f}'.format(acc[13] * 100.), '%\t')
        print('truck        : {:.6f}'.format(acc[14] * 100.), '%\t')
        print('bus          : {:.6f}'.format(acc[15] * 100.), '%\t')
        print('train        : {:.6f}'.format(acc[16] * 100.), '%\t')
        print('motorcycle   : {:.6f}'.format(acc[17] * 100.), '%\t')
        print('bicycle      : {:.6f}'.format(acc[18] * 100.), '%\t')
        if len(acc) == 20:
            print('small obstacles: {:.6f}'.format(acc[19] * 100.), '%\t')

def print_iou_per_class(iou):
    print('-----------IOU Per Class-----------')
    if len(iou) == 3:
        print('unknown      : {:.6f}'.format(iou[0] * 100.), '%\t')
        print('road         : {:.6f}'.format(iou[1] * 100.), '%\t')
        print('obstacle     : {:.6f}'.format(iou[2] * 100.), '%\t')
        print('Mean IoU     : {:.6f}'.format((iou[0] + iou[1] + iou[2]) / 3 * 100.))
    elif len(iou) == 12:
        print('sky         : {:.6f}'.format(iou[0] * 100.), '%\t')
        print('building    : {:.6f}'.format(iou[1] * 100.), '%\t')
        print('pole        : {:.6f}'.format(iou[2] * 100.), '%\t')
        print('road        : {:.6f}'.format(iou[3] * 100.), '%\t')
        print('sidewalk    : {:.6f}'.format(iou[4] * 100.), '%\t')
        print('tree        : {:.6f}'.format(iou[5] * 100.), '%\t')
        print('sign symbol : {:.6f}'.format(iou[6] * 100.), '%\t')
        print('fence       : {:.6f}'.format(iou[7] * 100.), '%\t')
        print('car         : {:.6f}'.format(iou[8] * 100.), '%\t')
        print('pedestrian  : {:.6f}'.format(iou[9] * 100.), '%\t')
        print('bicyclist   : {:.6f}'.format(iou[10] * 100.), '%\t')
        print('void        : {:.6f}'.format(iou[11] * 100.), '%\t')
    else:
        print('road         : {:.6f}'.format(iou[0] * 100.), '%\t')
        print('sidewalk     : {:.6f}'.format(iou[1] * 100.), '%\t')
        print('building     : {:.6f}'.format(iou[2] * 100.), '%\t')
        print('wall         : {:.6f}'.format(iou[3] * 100.), '%\t')
        print('fence        : {:.6f}'.format(iou[4] * 100.), '%\t')
        print('pole         : {:.6f}'.format(iou[5] * 100.), '%\t')
        print('traffic light: {:.6f}'.format(iou[6] * 100.), '%\t')
        print('traffic sign : {:.6f}'.format(iou[7] * 100.), '%\t')
        print('vegetation   : {:.6f}'.format(iou[8] * 100.), '%\t')
        print('terrain      : {:.6f}'.format(iou[9] * 100.), '%\t')
        print('sky          : {:.6f}'.format(iou[10] * 100.), '%\t')
        print('person       : {:.6f}'.format(iou[11] * 100.), '%\t')
        print('rider        : {:.6f}'.format(iou[12] * 100.), '%\t')
        print('car          : {:.6f}'.format(iou[13] * 100.), '%\t')
        print('truck        : {:.6f}'.format(iou[14] * 100.), '%\t')
        print('bus          : {:.6f}'.format(iou[15] * 100.), '%\t')
        print('train        : {:.6f}'.format(iou[16] * 100.), '%\t')
        print('motorcycle   : {:.6f}'.format(iou[17] * 100.), '%\t')
        print('bicycle      : {:.6f}'.format(iou[18] * 100.), '%\t')
        if len(iou) == 20:
            print('small obstacles: {:.6f}'.format(iou[19] * 100.), '%\t')

    
def print_precision_per_class(precision):
    print('-----------Precision Per Class-----------')
    if len(precision) == 3:
        print('unknown      : {:.6f}'.format(precision[0] * 100.), '%\t')
        print('road         : {:.6f}'.format(precision[1] * 100.), '%\t')
        print('obstacle     : {:.6f}'.format(precision[2] * 100.), '%\t')
        print('Mean Precision: {:.6f}'.format((precision[0] + precision[1] + precision[2]) / 3 * 100.))
    else:
        print('road         : {:.6f}'.format(precision[0] * 100.), '%\t')
        print('sidewalk     : {:.6f}'.format(precision[1] * 100.), '%\t')
        print('building     : {:.6f}'.format(precision[2] * 100.), '%\t')
        print('wall         : {:.6f}'.format(precision[3] * 100.), '%\t')
        print('fence        : {:.6f}'.format(precision[4] * 100.), '%\t')
        print('pole         : {:.6f}'.format(precision[5] * 100.), '%\t')
        print('traffic light: {:.6f}'.format(precision[6] * 100.), '%\t')
        print('traffic sign : {:.6f}'.format(precision[7] * 100.), '%\t')
        print('vegetation   : {:.6f}'.format(precision[8] * 100.), '%\t')
        print('terrain      : {:.6f}'.format(precision[9] * 100.), '%\t')
        print('sky          : {:.6f}'.format(precision[10] * 100.), '%\t')
        print('person       : {:.6f}'.format(precision[11] * 100.), '%\t')
        print('rider        : {:.6f}'.format(precision[12] * 100.), '%\t')
        print('car          : {:.6f}'.format(precision[13] * 100.), '%\t')
        print('truck        : {:.6f}'.format(precision[14] * 100.), '%\t')
        print('bus          : {:.6f}'.format(precision[15] * 100.), '%\t')
        print('train        : {:.6f}'.format(precision[16] * 100.), '%\t')
        print('motorcycle   : {:.6f}'.format(precision[17] * 100.), '%\t')
        print('bicycle      : {:.6f}'.format(precision[18] * 100.), '%\t')
        if len(precision) == 20:
            print('small obstacles: {:.6f}'.format(precision[19] * 100.), '%\t')


def print_recall_per_class(recall):
    print('-----------Recall Per Class-----------')
    if len(recall) == 3:
        print('unknown      : {:.6f}'.format(recall[0] * 100.), '%\t')
        print('road         : {:.6f}'.format(recall[1] * 100.), '%\t')
        print('obstacle     : {:.6f}'.format(recall[2] * 100.), '%\t')
        print('Mean Recall  : {:.6f}'.format((recall[0] + recall[1] + recall[2]) / 3 * 100.))
    else:
        print('road         : {:.6f}'.format(recall[0] * 100.), '%\t')
        print('sidewalk     : {:.6f}'.format(recall[1] * 100.), '%\t')
        print('building     : {:.6f}'.format(recall[2] * 100.), '%\t')
        print('wall         : {:.6f}'.format(recall[3] * 100.), '%\t')
        print('fence        : {:.6f}'.format(recall[4] * 100.), '%\t')
        print('pole         : {:.6f}'.format(recall[5] * 100.), '%\t')
        print('traffic light: {:.6f}'.format(recall[6] * 100.), '%\t')
        print('traffic sign : {:.6f}'.format(recall[7] * 100.), '%\t')
        print('vegetation   : {:.6f}'.format(recall[8] * 100.), '%\t')
        print('terrain      : {:.6f}'.format(recall[9] * 100.), '%\t')
        print('sky          : {:.6f}'.format(recall[10] * 100.), '%\t')
        print('person       : {:.6f}'.format(recall[11] * 100.), '%\t')
        print('rider        : {:.6f}'.format(recall[12] * 100.), '%\t')
        print('car          : {:.6f}'.format(recall[13] * 100.), '%\t')
        print('truck        : {:.6f}'.format(recall[14] * 100.), '%\t')
        print('bus          : {:.6f}'.format(recall[15] * 100.), '%\t')
        print('train        : {:.6f}'.format(recall[16] * 100.), '%\t')
        print('motorcycle   : {:.6f}'.format(recall[17] * 100.), '%\t')
        print('bicycle      : {:.6f}'.format(recall[18] * 100.), '%\t')
        if len(recall) == 20:
            print('small obstacles: {:.6f}'.format(recall[19] * 100.), '%\t')


# helper loading function that can be used by subclasses
def load_network(network, loading_epoch, save_dir=''):
    save_filename = '%s_model.pth' % (loading_epoch)
    save_path = os.path.join(save_dir, save_filename)        
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
    else:
        #network.load_state_dict(torch.load(save_path))
        try:
            # print torch.load(save_path).keys()
            # print network.state_dict()['Scale.features.conv2_1_depthconvweight']
            network.load_state_dict(torch.load(save_path, map_location='cuda:0'))
        except:   
            pretrained_dict = torch.load(save_path, map_location='cuda:0')               
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)
                print('Pretrained network has excessive layers; Only loading layers that are used' )
            except:
                print('Pretrained network has fewer layers; The following are not initialized:' )
                # from sets import Set
                # not_initialized = Set()
                for k, v in pretrained_dict.items():                      
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                not_initialized=[]
                # print(pretrained_dict.keys())
                # print(model_dict.keys())
                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized+=[k]#[k.split('.')[0]]
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)
    return network