import torch
import torchvision
from torch import nn


class SemanticImageExtractor(nn.Module):
    """
    This class expected image as input with size (64x64x3)
    """

    def __init__(self, output_class_num, feature_size=200):
        super(SemanticImageExtractor, self).__init__()
        self.alx_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.alx_layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.alx_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img

        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )

        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )

        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, x):
        x = self.alx_layer1(x)
        x = self.alx_layer2(x)
        x = self.alx_layer3(x)
        x = self.alx_layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label


class SemanticImageExtractorV2(nn.Module):
    """
    This class expected image as input with size (64x64x3)
    """

    def __init__(self, output_class_num, feature_size=200, pretrain=False):
        self.feature_size = feature_size
        self.num_classes = output_class_num
        super(SemanticImageExtractorV2, self).__init__()
        self.features = nn.Sequential(
            # Alex1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Alex3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Alex5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # return the same number of features but change width and height of img
        if (pretrain):
            import torchvision
            ori_alex = torchvision.models.alexnet(pretrained=True)
            ori_weight = ori_alex.state_dict()
            ori_weight.pop('classifier.1.weight')
            ori_weight.pop('classifier.1.bias')
            ori_weight.pop('classifier.4.weight')
            ori_weight.pop('classifier.4.bias')
            ori_weight.pop('classifier.6.weight')
            ori_weight.pop('classifier.6.bias')
            self.load_state_dict(ori_weight)
            del (ori_alex)
            del (ori_weight)

        self._add_classifier(self.num_classes, self.feature_size)

    def _add_classifier(self, num_classes, feature_size):
        self.fc06 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )
        self.fc07 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size),
            nn.ReLU()
        )
        self.fc08 = nn.Sequential(
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc06(x)
        semantic_features = self.fc07(x)
        p_label = self.fc08(semantic_features)
        return semantic_features, p_label


class SemanticEEGExtractor(nn.Module):
    def __init__(self, expected_shape: torch.Tensor, output_class_num: int, feature_size=200):
        """
        expected_shape [Batch_size, eeg_features, eeg_channel, sample_len]
        """
        super(SemanticEEGExtractor, self).__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=1)

        self.fc01 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(expected_shape.shape[1] * expected_shape.shape[2], 4096),
            nn.LeakyReLU()
        )

        self.fc02 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, feature_size)
        )

        self.fc02_act = nn.Tanh()

        self.fc03 = nn.Sequential(
            nn.Linear(feature_size, output_class_num),
            nn.Softmax())

    def forward(self, eeg: torch.Tensor):
        eeg = eeg.unsqueeze(1)
        x = self.batch_norm(eeg)
        if torch.isnan(self.batch_norm.weight).item():
            print("<X> : NAN detected in batch_norm weight")
            exit()
        x = x.reshape([x.shape[0], -1])
        x = self.fc01(x)
        semantic_features = self.fc02(x)
        x = self.fc02_act(semantic_features)
        label = self.fc03(x)
        return semantic_features, label


# Generator Code

# default ngf = 64
class ChakkyGenerator(nn.Module):
    EXPECTED_NOISE = 50

    def __init__(self, input_size):
        super(ChakkyGenerator, self).__init__()
        nz = input_size
        ngf = 64
        nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #                   in_chan, out_chan, krn_size, stride, pad
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # From 64 to 3
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, semantic, label):
        tnsr = torch.cat((z, semantic, label), 1)
        new_shape = z.shape[1] + semantic.shape[1] + label.shape[1]
        tnsr = tnsr.reshape(tnsr.shape[0], new_shape, 1, 1)
        return self.main(tnsr)


class Generator(nn.Module):  # <<- CGAN
    # How can we input both label and features?
    # EXPECTED_NOISE = 2064  # << For EEGImageNet with 48x48
    # EXPECTED_NOISE = 2098  # << For VeryNiceDataset with 48x48
    # EXPECTED_NOISE = 100  # For now, I will use this noise size
    # EXPECTED_NOISE = 2101  # Cylinder_RGB with 48x48

    def __init__(self, latent_size, num_classes, embed_size):
        super(Generator, self).__init__()
        self.lt_s = latent_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        l1_size = self.lt_s + self.embed_size
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(l1_size, 128, kernel_size=5, stride=2, output_padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, output_padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.Tanh()
        )
        # self.deconv5 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=1, output_padding=1, bias=False),
        #     nn.Tanh()
        # )

        self.embedder = nn.Embedding(num_classes, embedding_dim=embed_size)

    def __forward_check(self, eeg_semantic, label):
        tag = "<Generator> : "
        if eeg_semantic.shape[1] != self.lt_s:
            raise RuntimeError(tag + "Incorrect shape of vector \'eeg_semantic\'")

    # -- Expected shape --
    # z.shape = (3839,)
    # eeg_semantic.shape = (200,)
    # label.shape = (10,)
    def forward(self, semantic, label):
        """
        :param semantic: The latent vector
        :param label: Label tensor (Expected as an index format (Single digit format))
        :return:
        """
        self.__forward_check(semantic, label)
        x = torch.cat((semantic, self.embedder(label)), 1)
        x = x.reshape(x.shape[0], 200 + self.embed_size, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        # x = self.deconv5(x)  # shape >>
        return x


# Should D1 and D2 takes an real/gen image as an input?
# D1 : Image only
# D2 : Semantic features and label
class D1(nn.Module):
    def __init__(self):
        super(D1, self).__init__()
        self.conv1 = nn.Sequential(  # Currently we input black and white img
            # nn.BatchNorm2d(num_features=6),
            nn.Conv2d(6, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.InstanceNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5),
            # nn.InstanceNorm2d(num_features=512, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=46),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=46, out_features=1),
            nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(x):
        shape = (x.shape[1], x.shape[2], x.shape[3])
        if shape != (3, 64, 64):
            raise RuntimeError("Expected shape", (3, 64, 64))

    def forward(self, x1, x2):
        '''

        :param x1: First image to input
        :param x2: Second image to input but... How we gonna concat? concat in channel dim? YES I THINK WE CAN!
        :return: real or not
        '''
        self.__forward_check(x1)
        self.__forward_check(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        x = self.final_fc(x)
        return x


# In the paper, This is D1
class D2(nn.Module):
    def __init__(self, num_classes, embed_size):
        super(D2, self).__init__()

        self.embedder = nn.Embedding(num_classes, embedding_dim=embed_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(num_features=64, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.InstanceNorm2d(num_features=128, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5),
            # nn.InstanceNorm2d(num_features=256, affine=True),
            nn.LeakyReLU(0.2)
        )

        # self.final_fc = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(12784, 226),  # TODO: Check the shape
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(226, 1)
        # )

        # self.final_fc_verynice = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(12750, 226),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(226, 1)
        # )

        # self.final_fc_cylinder = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(12747, 226),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(226, 1),
        #     nn.Sigmoid()
        # )

        self.final_fc = nn.Sequential(
            nn.Linear(496, 46),
            nn.LeakyReLU(),
            nn.Linear(46, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def __forward_check(img):  # , eeg_features, eeg_label):
        # if eeg_features.shape[1] != 200:
        #     raise RuntimeError("Expected features size = 200")
        # if eeg_label.shape[1] != 569:
        #     raise RuntimeError("Expected shape size = 569")
        img_shape = (img.shape[1], img.shape[2], img.shape[3])
        if img_shape != (3, 64, 64):
            raise RuntimeError("Expected shape", (3, 64, 64))

    def forward(self, img, features, label):  # , eeg_features, eeg_label):
        """
        :param img: Stimuli image
        :param features: Latent vector
        :param label: Label (Expected as single digit)
        :return:
        """
        # self.__forward_check(img, eeg_features, eeg_label)
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)

        x = x.flatten(start_dim=1)
        x = torch.cat((x, features), 1)  # Concat eeg_features
        x = torch.cat((x, self.embedder(label)), 1)  # Concat label
        x = self.final_fc(x)
        # x = self.final_fc_cylinder(x)
        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        # self.ngpu = ngpu
        dis_dim = 64
        num_channel = 3
        self.net = nn.Sequential(
            # no batch norm in the first layer
            # Input: batch x num_channel x 64 x 64
            nn.Conv2d(
                num_channel, dis_dim, kernel_size=4, stride=2, padding=1,
            ),  # batch x 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(dis_dim, dis_dim * 2, 4, 2, 1),  # batch x 128 x 16 x 16
            self._block(dis_dim * 2, dis_dim * 4, 4, 2, 1),  # batch x 256 x 8 x 8
            self._block(dis_dim * 4, dis_dim * 8, 4, 2, 1),  # batch x 512 x 4  x 4
            nn.Conv2d(dis_dim * 8, 1, kernel_size=4, stride=2, padding=0),  # batch x 1 x 1 x 1 for classification
            #             nn.Sigmoid(), #<------removed!
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # <----changed here
            nn.LeakyReLU(0.2, True)  # slope = 0.2, in_place = True
        )

    def forward(self, x):
        return self.net(x)


def __test_execution():
    sample = torch.rand(1, 3, 64, 64)
    model = D2()
    out = model(sample)
