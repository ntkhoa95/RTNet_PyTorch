import cv2, onnx
import torch, os, onnxruntime
from models.RTFNet import RTFNet
from utils.util import *
import depthai as dai

if __name__ == "__main__":
    onnx_file = "rtfnet.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_file)

    # Set up camera
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

            if latest_packet["rgb"] is not None:
                frame_rgb = latest_packet["rgb"].getCvFrame()
                
            if latest_packet["depth"] is not None:
                frame_depth = latest_packet["depth"].getFrame()

                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

                # frame_depth = latest_packet['depth'].getFrame()
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
                frame = torch.FloatTensor(frame).unsqueeze(0)

                logits = ort_session.run(None, {'input': frame.numpy()})[0].argmax(1)
                pred = logits.squeeze(0)
                pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                
                for cid in range(len(palette)):
                    pred_img[pred == cid] = palette[cid]

                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                pred_img = cv2.addWeighted(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), 0.5, pred_img, 0.5, 0)
                pred_img = cv2.resize(pred_img, (1280, 720))
                cv2.imshow("frame", pred_img)

            if cv2.waitKey(1) == ord('q'):
                break