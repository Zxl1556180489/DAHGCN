import time
import os


class Config(object):
    """
    class to save config parameter
    应用程序需要某种形式的配置。你可能会需要根据应用环境更改不同的设置，
    比如开关调试模式、 设置密钥、或是别的设定环境的东西。
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280  # input image size  VD数据集排在前面，输入尺寸720*1280
        self.batch_size = 32  # train batch size 训练批量32
        self.test_batch_size = 8  # test batch size  测试批量8
        self.num_boxes = 12  # max number of bounding boxes in each frame  VD数据集总共12个运动员

        # self.in_dim = in_dim
        # self.temporal_pooled_first = temporal_pooled_first

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = True
        self.device_list = "0,1"  # id list of gpus used for training
        
        # Dataset
        assert(dataset_name in ['volleyball', 'collective'])
        self.dataset_name = dataset_name
        
        if dataset_name=='volleyball':  # 排球数据集  # 96 129 /home/hi/data/volleyball/videos/  403 /home/hi/data/volleyball/
            self.data_path='/home/hi/datasets/volleyball/'  # data path for the volleyball dataset 数据集路径
            # 训练视频序列号
            self.train_seqs = [1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]  # video id list of train set
            # 测试视频序列号
            self.test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]  # video id list of test set
            # [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]
            
        else:
            self.data_path = '/home/hi/datasets/collective/'  # data path for the collective dataset
            # 测试视频序列号
            self.test_seqs = [5,6,7,8,9,10,11,15,16,25,28,29]
            # 训练视频序列号
            self.train_seqs = [s for s in range(1,45) if s not in self.test_seqs]
            # if s not in self.test_seqs 45个视频 1-44
        
        # Backbone 
        self.backbone = 'inv3'  # 可以改 inv3 vgg16 vgg19   不同骨干网络进行分析模型性能
        self.crop_size = 5, 5  # crop size of roi align  roi align后的输出大小为5x5。
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        # 排球数据集输入为720*1280，输出为87*157； 集体行为数据集输入为480, 720，输出为57*87
        self.out_size = 87, 157  # output feature map size of backbone
        # embedding 使用线性和非线性转换对复杂的数据进行自动特征抽取  映射成1056维
        self.emb_features = 1056   # output feature map channel of backbone

        
        # Activity Action
        self.num_actions = 9  # number of action categories
        self.num_activities = 8  # number of activity categories
        # 用于平衡个人行为损失和群体行为损失
        self.actions_loss_weight = 1.0  # 1.0 # weight used to balance action loss and activity loss
        # 个人行为权重
        self.actions_weights = None

        # Sample
        self.num_frames = 1  # 3
        # 采样间隔10帧 前5帧 后4帧  5+4=9
        self.num_before = 5
        self.num_after = 4

        # GCN
        self.num_features_boxes = 1024  # 检测框特征数   最后inv3 特征提取的维度
        self.num_features_relation = 256  # 关系特征数量 论文的dk
        self.num_graph = 16  # number of graphs 通过改变图数量（1 2 4 8 16 32） 进行 分析模型性能
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1  # number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  # distance mask threshold in position relation  预先设定的阈值  1/5

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  # initial learning rate  初始学习率
        # 学习率随epoch改变
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  # change learning rate in these epochs
        # 一个深层网络的早期drop的概率会比较小，在0.1~0.3左右 经过交叉验证,隐含节点dropout率等于0.5的时候效果最好,原因是0.5的时候dropout随机生成的网络结构最多。
        self.train_dropout_prob = 0.3  # 0.3 dropout probability
        self.weight_decay = 0  # l2 weight decay
    
        self.max_epoch = 150  # max training epoch
        # 测试epoch间隔为2
        self.test_interval_epoch = 2
        
        # Exp
        self.training_stage=1  # specify stage1 or stage2
        # 第一阶段采用基本模型，第二阶段需要设置
        self.stage1_model_path = '/home/hi/Conrad/1/GAR-2019/result/[Volleyball_stage1_inv3_stage1]<2021-12-03_17-03-06>/stage1_epoch134_89.90%.pth'  # path of the base model, need to be set in stage2
        # 训练完成最优模型
        self.stage2_model_path = '/home/hi/data/Conrad/1/GAR-2019/result/[Volleyball_stage2_inv3_stage2]<2022-07-10_15-34-21>/stage2_epoch134_94.10%.pth'  # path of the gcn model, need to be set in stage3
        self.test_before_train=False
        self.exp_note='Group-Activity-Recognition'  # 在scripts文件夹设置超参数 则可以改变这个cofig参数
        self.exp_name=None
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:  # 结果保存文件夹名
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_%s_stage%d]<%s>'%(self.exp_note,self.backbone,self.training_stage,time_str)
            
        self.result_path='result/%s' % self.exp_name  # 结果路径result/%s
        self.log_path='result/%s/log.txt' % self.exp_name  # 训练日志
            
        if need_new_folder:  # 存在文件夹 则创建一个新文件夹
            os.mkdir(self.result_path)