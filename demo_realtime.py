import os, torch, cv2
from sklearn import pipeline
import numpy as np
import stat, argparse
from tqdm import tqdm
import depthai as dai

import torch.nn.functional as F
from torch.autograd import Variable
from datasets.GMRPD_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from utils.util import *
from models import RTFNet

from utils.util import compute_results, visualise, SegMetrics

import warnings
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

    # Create pipeline
    down_scale_flag = True
    pipe_line = dai.Pipeline()
    queue_name = []

    # Define sources and outputs
    cam_rgb = pipe_line.createColorCamera()
    left    = pipe_line.createMonoCamera()
    right   = pipe_line.createMonoCamera()
    stereo  = pipe_line.createStereoDepth()

    rgb_out = pipe_line.createXLinkOut()
    depth_out = pipe_line.createXLinkOut()

    rgb_out.setStreamName('rgb')
    queue_name.append('rgb')
    depth_out.setStreamName('depth')
    queue_name.append('depth')

    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    if down_scale_flag: cam_rgb.setIspScale(2, 3)

    cam_rgb.initialControl.setManualFocus(130)

    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    left.setFps(30)
    right.setFps(30)

    stereo.setConfidenceThreshold(245)
    stereo.setRectifyEdgeFillColor(0)

    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    cam_rgb.isp.link(rgb_out.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depth_out.input)

    # Connect to device and start pipeline
    with dai.Device(pipe_line) as device:
        device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
        device.getOutputQueue(name='depth', maxSize=4, blocking=False)

        while True:
            latest_packet = {}
            latest_packet['rgb'] = None
            latest_packet['depth'] = None

            queue_event = device.getQueueEvents(('rgb', 'depth'))
            for queue_name in queue_event:
                packets = device.getOutputQueue(queue_name).tryGetAll()
                if len(packets) > 0:
                    latest_packet[queue_name] = packets[-1]

            if latest_packet['rgb'] is not None and latest_packet['depth'] is not None:
                frame_rgb = latest_packet['rgb'].getCvFrame()
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

                frame_depth = latest_packet['depth'].getFrame()
                frame_depth_u8 = cv2.normalize(frame_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                frame_depth_u8 = cv2.resize(frame_depth_u8, (640, 480))

                unknown_area   = [0, 0, 255]
                drivable_area  = [0, 255, 0]
                road_anomalies = [255, 0, 0]
                palette = np.array([unknown_area, drivable_area, road_anomalies]).tolist()

                frame_depth_u8 = frame_depth_u8[:, :, np.newaxis]
                frame = np.dstack((frame_rgb, frame_depth_u8))

                frame = frame.astype('float32')
                frame = np.transpose(frame, (2, 0, 1)) / 255.0
                frame = torch.FloatTensor(frame).unsqueeze(0).cuda(args.gpu_ids)

                logits = model(frame).argmax(1)
                pred = logits.squeeze(0).cpu().numpy()
                pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for cid in range(len(palette)):
                    pred_img[pred == cid] = palette[cid]

                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("frame", pred_img)

            if cv2.waitKey(1) == ord('q'):
                break

        