from .Piplines import *
class Activity_Level(Piplines):  # 繼承父類Piplines.py
    """docstring for Action_Level"""
    def __init__(self, dataset_root, dataset_name, mode):  # 规定“activity” stage
        super(Activity_Level, self).__init__(dataset_root, dataset_name, 'activity', mode)

        
    def extractFeas(self, save_folder):
        pass
        print('Done, the features files are saved at ' + save_folder + '\n')

    # {'trainval': 3493, 'test': 1337}
    def loadModel(self, pretrained=False):  # 加载模型HiGCIN  pretrained=False
        net = Models.HiGCIN(pretrained, self.dataset_name, model_confs=self.model_confs, mode=self.mode)
        return net

    def loss(self):
        pass