import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


#output an attention map(1*H*W)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)#softmax by rows
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size() # [N, C, H ,W]
        if self.pooling_type == 'att':
            input_x = x # [N, C, H ,W]
            input_x = input_x.view(batch, channel, height * width) # [N, C, H * W]
            input_x = input_x.unsqueeze(1) # [N, 1, C, H * W]
            context_mask = self.conv_mask(x) # [N, 1, H, W]
            context_mask = context_mask.view(batch, 1, height * width) # [N, 1, H * W]
            context_mask = self.softmax(context_mask) #softmax the "H*W"!!!!
            context_mask = context_mask.unsqueeze(-1) # [N, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask) # ([N, 1, C, H * W] X [N, 1, H * W, 1]-->)[N, 1, C, 1]
            context = context.view(batch, channel, 1, 1) # [N, C, 1, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)# [N, C, 1, 1]
        out = x # [N, C, H ,W]
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out
    
# for conv5
class ACCoM5(nn.Module):
    def __init__(self, cur_channel):
        super(ACCoM5, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.context = ContextBlock(cur_channel, 0.5)

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

    def forward(self, x_pre, x_cur):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur + x_cur_1)
        x_cur_3 = self.cur_b3(x_cur + x_cur_2)
        x_cur_4 = self.cur_b4(x_cur + x_cur_3)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_context = self.context(x_cur_all)

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))
        x_LocAndGlo = cur_context + pre_sa + x_cur

        return x_LocAndGlo

class ACCoM(nn.Module):
    def __init__(self, cur_channel):
        super(ACCoM, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)
        self.context = ContextBlock(cur_channel, 0.5)

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur + x_cur_1)
        x_cur_3 = self.cur_b3(x_cur + x_cur_2)
        x_cur_4 = self.cur_b4(x_cur + x_cur_3)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_context = self.context(x_cur_all)

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur_all.mul(self.lat_sa(x_lat))
        x_LocAndGlo = cur_context + pre_sa + lat_sa + x_cur

        return x_LocAndGlo



class SE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SE, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
    def forward(self, x):
        x = self.convert(x) #channel = 384,192,96,48
        return x

class HGDModule_4(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(HGDModule_4, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 2, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels))
            # norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.ewai = nn.Conv2d(out_channels, 384, 3, padding=1, bias=False)
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, guide2):
        n, c, h, w = x.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)
        x_cat = torch.cat([guide2_down, x], 1)
        guide_cat = torch.cat([guide2, x_up0], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h * w)
        # f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)
        f_center_norm = self.norm_center(f_center_norm)
        # n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat_conv = self.conv_affinity0(guide_cat)
        guide_cat_value_avg = guide_cat_conv + value_avg
        f_affinity = self.conv_affinity1(guide_cat_value_avg)
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        # x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up = norm_aff * x_center.bmm(f_affinity)
        x_up = x_up.view(n, self.out_channels, h_aff, w_aff)
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)
        x_up_conv = self.conv_up(x_up_cat)
        out = self.ewai(x_up_conv)
        return out

class HGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(HGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center = nn.Sequential(
            nn.Conv2d(in_channels * 3, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels))
            # norm_layer(center_channels))
        self.norm_center = nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            nn.BatchNorm2d(center_channels),
            # norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.ewai = nn.Conv2d(out_channels, 384, 3, padding=1, bias=False)
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()
        n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)
        x_cat = torch.cat([guide2_down, guide1_down, x], 1)
        guide_cat = torch.cat([guide2, x_up1, x_up0], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h * w)
        f_center_norm = f_center.view(n, self.center_channels, h * w)
        f_center_norm = self.norm_center(f_center_norm)
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        ###################################
        # f_affinity = self.conv_affinity(guide_cat)
        guide_cat_conv = self.conv_affinity0(guide_cat)
        guide_cat_value_avg = guide_cat_conv + value_avg
        f_affinity = self.conv_affinity1(guide_cat_value_avg)
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        x_up = norm_aff * x_center.bmm(f_affinity)
        x_up = x_up.view(n, self.out_channels, h_aff, w_aff)
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)
        x_up_conv = self.conv_up(x_up_cat)
        out = self.ewai(x_up_conv)
        return out

class DenseModel_work2(nn.Module):
    def __init__(self):
        super(DenseModel_work2, self).__init__()

        self.dense = models.densenet161(pretrained=True).features

        for param in self.dense.parameters():
            param.requires_grad = True
        self.conv_layer0 = nn.Sequential(*list(self.dense)[:3])
        self.conv_layer1 = nn.Sequential(
           self.dense.pool0,
           self.dense.denseblock1,
           *list(self.dense.transition1)[:3]
        )
        self.conv_layer2 = nn.Sequential(
           self.dense.transition1[3],
           self.dense.denseblock2,
           *list(self.dense.transition2)[:3]
        )
        self.conv_layer3 = nn.Sequential(
           self.dense.transition2[3],
           self.dense.denseblock3,
           *list(self.dense.transition3)[:3]
        )
        self.conv_layer4 = nn.Sequential(
           self.dense.transition3[3],
           self.dense.denseblock4
        )
        self.convert5 = SE(2208, 384)
        self.convert4 = SE(1056, 384)
        self.convert3 = SE(384, 384)
        
        self.ACCoM5 = ACCoM5(384)
        self.ACCoM4 = ACCoM(384)
        self.ACCoM3 = ACCoM(384)
        

        self.trytry = HGDModule(384,600,768)
        self.trytry_4 = HGDModule_4(384, 600, 768)
        self.up_module_5 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up_module_4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up_module_3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.readout_layer = nn.Sequential(
            nn.Conv2d(in_channels=384*3, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, bias=True),

        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x1 = self.conv_layer0(x)
        x2 = self.conv_layer1(x1)
        x3 = self.conv_layer2(x2)
        x4 = self.conv_layer3(x3)
        x5 = self.conv_layer4(x4)
        x_size = x.size()[2:]


        x5 = self.convert5(x5 )
        x4 = self.convert4(x4)
        x3 = self.convert3(x3)


        y5 = self.ACCoM5(x4, x5)
        y4_1 = self.ACCoM4(x3, x4, x5)
        y3 = self.ACCoM3(x2, x3, x4)


        y4 = self.trytry_4(y5, y4_1)
        y3 = self.trytry(y5,y4_1,y3)



        y5_fusion = self.up_module_5(y5)
        y4_fusion = self.up_module_4(y4)
        y3_fusion = self.up_module_3(y3)
        y2 = torch.cat((y5_fusion, y4_fusion, y3_fusion),1)
        y1 = self.readout_layer(y2)

        s1 = self.sig(y1)
        score1 = F.interpolate(s1, x_size, mode='bilinear', align_corners=True)

        score1 = score1.squeeze(1)
        return score1