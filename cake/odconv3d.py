import torch
import torch.nn as nn
import torch.nn.functional as F

def to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    return x

class Attention3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention3D, self).__init__()
        # FIX 1: Xử lý kernel_size dạng tuple (kt, kh, kw)
        self.kernel_size = to_3tuple(kernel_size) 
        self.kernel_num = kernel_num
        self.temperature = 1.0

        attention_channel = max(int(in_planes * reduction), min_channel)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm3d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        # FIX 2: Logic cho Spatial Attention với kernel bất đối xứng
        kt, kh, kw = self.kernel_size
        if kt * kh * kw == 1: # 1x1x1 kernel
            self.func_spatial = self.skip
        else:
            # Output channels bằng tổng số phần tử trong kernel
            self.spatial_fc = nn.Conv3d(attention_channel, kt * kh * kw, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        # FIX 3: Sửa view để support broadcasting đúng với weight (B, 1, 1, 1, Kt, Kh, Kw)
        # Weight shape gốc: (1, n, Out, In, Kt, Kh, Kw) -> 7 dims
        kt, kh, kw = self.kernel_size
        sp = self.spatial_fc(x)
        return torch.sigmoid(sp.view(x.size(0), 1, 1, 1, kt, kh, kw) / self.temperature)

    def get_kernel_attention(self, x):
        # Output: (B, n, 1, 1, 1, 1, 1) -> Khớp 7 dims
        return F.softmax(self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1, 1) / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv3d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        # FIX 4: Convert kernel_size to tuple
        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        
        self.groups = groups
        self.kernel_num = kernel_num
        
        self.attention = Attention3D(in_planes, out_planes, kernel_size, groups=groups,
                                     reduction=reduction, kernel_num=kernel_num)
        
        # FIX 5: Parameter shape sử dụng tuple (Kt, Kh, Kw)
        kt, kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kt, kh, kw),
                                   requires_grad=True)
        self._initialize_weights()

        if kt * kh * kw == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        batch_size, in_planes, time, height, width = x.size()
        kt, kh, kw = self.kernel_size
        
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        
        x = x * channel_attention
        x = x.reshape(1, -1, time, height, width) 

        # weight: (n, Out, In, Kt, Kh, Kw) -> (1, n, Out, In, Kt, Kh, Kw)
        weight_reshaped = self.weight.unsqueeze(0)
        
        # FIX 6: Broadcasting logic đảm bảo đúng dimensions
        # spatial: (B, 1, 1, 1, Kt, Kh, Kw)
        # kernel:  (B, n, 1, 1, 1, 1,  1)
        # weight:  (1, n, Out,In, Kt, Kh, Kw)
        aggregate_weight = spatial_attention * kernel_attention * weight_reshaped
        
        if self.kernel_num > 1:
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
        else:
            aggregate_weight = aggregate_weight.view(batch_size, self.out_planes, self.in_planes // self.groups, kt, kh, kw)
        
        # Reshape cho Group Conv: (B*Out, In/g, Kt, Kh, Kw)
        aggregate_weight = aggregate_weight.reshape(
            batch_size * self.out_planes, 
            self.in_planes // self.groups, 
            kt, kh, kw
        )
        
        output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        
        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        # Implementation cho 1x1x1 giữ nguyên, nhưng cần chỉnh weight squeeze nếu cần
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv3d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)