"""
	Common Piplines
"""
from abc import ABCMeta, abstractmethod
from torchvision import transforms
import torch
import Configs
import Data
import Models
import Solver
# import os
# os.environ['CUDA_VISIBLE_DEVICE'] = '1'
# device_ids = [1]

class Piplines(object):
    """docstring for Piplines"""

    def __init__(self, dataset_root, dataset_name, stage, mode):
        super(Piplines, self).__init__()
        self.dataset_root = dataset_root  # '/home/hi/data/volleyball/'
        self.dataset_name = dataset_name  # 'VD'
        self.stage = stage  # 'activity'
        self.mode = mode  # 'end_to_end'
        self.configuring()
        
    def configuring(self):
        # Dataset configs: 数据集配置
        self.data_confs = Configs.Data_Configs(
            self.dataset_root, self.dataset_name, self.stage, self.mode).configuring()
        print('data_confs', self.data_confs)
        
        # Model configs: 模型配置 数据集名称 stage
        self.model_confs = Configs.Model_Configs(
            self.dataset_name, self.stage).configuring()
        
        self.data_loaders, self.data_sizes = self.loadData()  # 加载数据
        self.net = self.loadModel()  # 加载模型  HiCGIN

        if torch.cuda.is_available():
            # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 单GPU或者CPU
            # self.net.to(device)
            self.net = self.net.cuda('cuda:1')  # 单卡 使用.cuda将计算或者数据从CPU移动至GPU  在CPU上进行运算时，比如使用plt可视化绘图, 我们可以使用.cpu将计算或者数据转移至CPU.
            # self.net = torch.nn.DataParallel(self.net).cuda()  # 并行运算
        print(self.net)  # 输出当前模型网络结构
       

        # Solver configs:
        self.solver_confs = Configs.Solver_Configs(self.dataset_name, self.data_loaders, self.data_sizes, self.net, self.stage, self.mode, self.data_confs).configuring()
        print('solver_confs', self.solver_confs)

        self.solver = Solver.Solver(self.net, self.model_confs, self.solver_confs)

    def loadModel(self, model_confs):
        raise NotImplementedError

    def defineLoss(self, model_confs):
        raise NotImplementedError

    def trainval(self):
        self.solver.train_model()

    def test(self):
        self.solver.test_model()

    def loadData(self, phases=['trainval', 'test']):  # 训练验证集 or 测试集
        if self.data_confs.data_type == 'img':
            data_transforms = {
                'trainval': transforms.Compose([
                    transforms.Resize((224, 224)),  # 299,299 for inception 把图片缩放到 (224, 224) 大小 (下面的所有操作都是基于缩放之后的图片进行的)，然后再进行其他 transform 操作。
#                     transforms.Resize((299, 299)), # 299,299 for inception
                    #transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize((224, 224)),
#                     transforms.Resize((299, 299)), # 299,299 for inception
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        else:
            data_transforms = None
        # 设置dataset data_loaders  进入VD_img.py
        dataset = {phase: eval('Data.' + self.dataset_name + '_' + self.data_confs.data_type)(
            self.data_confs, self.model_confs, phase, data_transforms[phase] if data_transforms else None) for phase in phases}
        data_loaders = {phase: torch.utils.data.DataLoader(dataset[phase],batch_size=self.data_confs.batch_size[phase], num_workers=8, shuffle=True) for phase in phases}
        # num_workers=8  计算trainval与test的帧数 3493 1337
        data_sizes = {phase: len(dataset[phase]) for phase in phases}  # trainval：3493 test：1337
        return data_loaders, data_sizes  # 返回数据加载器

    
    def evaluate(self, model_path=None):
        if model_path:
            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = torch.load('./weights/'+self.dataset_name+'/xxx.pth')
        self.net.load_state_dict(pretrained_dict)
        self.net.eval()
        self.solver.evaluate()