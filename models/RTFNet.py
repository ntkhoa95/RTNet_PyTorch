
import torch
import torch.nn as nn
import torchvision.models as models

class RTFNet(nn.Module):
    def __init__(self, n_class, num_resnet_layers=50, verbose=False):
        super(RTFNet, self).__init__()
        self.verbose = verbose
        self.n_class = n_class
        self.num_resnet_layers = num_resnet_layers

        if self.num_resnet_layers == 18:
            org_resnet_model_1 = models.resnet18(pretrained=True)
            org_resnet_model_2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            org_resnet_model_1 = models.resnet34(pretrained=True)
            org_resnet_model_2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            org_resnet_model_1 = models.resnet50(pretrained=True)
            org_resnet_model_2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            org_resnet_model_1 = models.resnet50(pretrained=True)
            org_resnet_model_2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            org_resnet_model_1 = models.resnet152(pretrained=True)
            org_resnet_model_2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        # RGB Encoder
        self.rgb_encoder_conv1     = org_resnet_model_1.conv1
        self.rgb_encoder_bn1       = org_resnet_model_1.bn1
        self.rgb_encoder_relu      = org_resnet_model_1.relu
        self.rgb_encoder_maxpool   = org_resnet_model_1.maxpool
        self.rgb_encoder_layer1    = org_resnet_model_1.layer1
        self.rgb_encoder_layer2    = org_resnet_model_1.layer2
        self.rgb_encoder_layer3    = org_resnet_model_1.layer3
        self.rgb_encoder_layer4    = org_resnet_model_1.layer4

        # Depth/Thermal Encoder
        self.depth_encoder_conv1   = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_encoder_conv1.weight.data = torch.unsqueeze(torch.mean(org_resnet_model_2.conv1.weight.data, dim=1), dim=1)
        self.depth_encoder_bn1     = org_resnet_model_2.bn1
        self.depth_encoder_relu    = org_resnet_model_2.relu
        self.depth_encoder_maxpool = org_resnet_model_2.maxpool
        self.depth_encoder_layer1  = org_resnet_model_2.layer1
        self.depth_encoder_layer2  = org_resnet_model_2.layer2
        self.depth_encoder_layer3  = org_resnet_model_2.layer3
        self.depth_encoder_layer4  = org_resnet_model_2.layer4

        # Decoder
        self.deconv1 = self._make_transpose_layer(block=TransBottleneck, planes=self.inplanes//2, blocks=2, stride=2)
        self.deconv2 = self._make_transpose_layer(block=TransBottleneck, planes=self.inplanes//2, blocks=2, stride=2)
        self.deconv3 = self._make_transpose_layer(block=TransBottleneck, planes=self.inplanes//2, blocks=2, stride=2)
        self.deconv4 = self._make_transpose_layer(block=TransBottleneck, planes=self.inplanes//2, blocks=2, stride=2)
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)

    
    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),\
                                        nn.BatchNorm2d(planes))
        elif self.inplanes != planes:
            upsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),\
                                        nn.BatchNorm2d(planes))

        for module in upsample.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        rgb = input[:, :3]
        depth = input[:, 3:]

        if self.verbose:
            print("RGB input size: ", rgb.size())            # 480x640
            print("Depth input size: ", depth.size())        # 480x640

        # ENCODER PART
        ######################################################################################

        rgb = self.rgb_encoder_conv1(rgb)                              # Conv Layer
        if self.verbose: print("RGB after Conv_1 size: ", rgb.size())       # 240x320
        rgb   = self.rgb_encoder_bn1(rgb)                              # BatchNorm Layer
        if self.verbose: print("RGB after Conv_1 size: ", rgb.size())       # 240x320
        rgb  = self.rgb_encoder_relu(rgb)                              # ReLU Layer
        if self.verbose: print("RGB after Conv_1 size: ", rgb.size())       # 240x320

        depth = self.depth_encoder_conv1(depth)                        # Conv Layer
        if self.verbose: print("Depth after Conv_1 size: ", depth.size())   # 240x320
        depth   = self.depth_encoder_bn1(depth)                        # BatchNorm Layer
        if self.verbose: print("Depth after Conv_1 size: ", depth.size())   # 240x320
        depth  = self.depth_encoder_relu(depth)                        # ReLU Layer
        if self.verbose: print("Depth after Conv_1 size: ", depth.size())   # 240x320

        ######################################################################################

        rgb = rgb + depth                                              # Fusion Layer 1
        if self.verbose: print("RGB after 1st Fusion size: ", rgb.size())

        rgb = self.rgb_encoder_maxpool(rgb)                            # Max Pooling Layer
        if self.verbose: print("RGB after MaxPool size: ", rgb.size())      # 120x160

        depth = self.depth_encoder_maxpool(depth)                      # Max Pooling Layer
        if self.verbose: print("Depth after MaxPool size: ", depth.size())  # 120x160

        rgb = self.rgb_encoder_layer1(rgb)                             # Residual 1st
        if self.verbose: print("RGB after 1st Residual size: ", rgb.size())
        depth = self.depth_encoder_layer1(depth)                       # Residual 1st
        if self.verbose: print("Depth after 1st Residual size: ", depth.size())

        ######################################################################################
        
        rgb = rgb + depth                                              # Fusion Layer 2
        if self.verbose: print("RGB after Fusion in block 2 size: ", rgb.size())

        rgb = self.rgb_encoder_layer2(rgb)                             # Residual 2nd
        if self.verbose: print("RGB after 2nd Residual size: ", rgb.size())
        depth = self.depth_encoder_layer2(depth)                       # Residual 2nd
        if self.verbose: print("Depth after 2nd Residual size: ", depth.size())

        ######################################################################################

        rgb = rgb + depth                                              # Fusion Layer 3
        if self.verbose: print("RGB after Fusion in block 3 size: ", rgb.size())

        rgb = self.rgb_encoder_layer3(rgb)                             # Residual 3th
        if self.verbose: print("RGB after Residual 3th size: ", rgb.size())
        depth = self.depth_encoder_layer3(depth)                       # Residual 3th
        if self.verbose: print("Depth after Residual 3th size: ", depth.size())

        ######################################################################################

        rgb = rgb + depth                                              # Fusion Layer 4
        if self.verbose: print("RGB after Fusion in block 4 size: ", rgb.size())

        rgb = self.rgb_encoder_layer4(rgb)                             # Residual 4th
        if self.verbose: print("RGB after Residual 4th size: ", rgb.size())
        depth = self.depth_encoder_layer4(depth)                       # Residual 4th
        if self.verbose: print("Depth after Residual 4th size: ", depth.size())

        fusion = rgb + depth                                              # Fusion Layer Final
        if self.verbose: print("RGB after final Fusion size: ", fusion.size())

        ######################################################################################
        

        # DENCODER PART
        ######################################################################################
        fusion = self.deconv1(fusion)
        fusion = self.deconv2(fusion)
        fusion = self.deconv3(fusion)
        fusion = self.deconv4(fusion)
        fusion = self.deconv5(fusion)

        return fusion


class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)
        return out

def main():
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 480, 640)
    depth = torch.randn(batch_size, 1, 480, 640)
    model = RTFNet(n_class=3, num_resnet_layers=18)
    inputs = torch.cat((rgb, depth), dim=1)
    model(inputs)


if __name__ == "__main__":
    main()