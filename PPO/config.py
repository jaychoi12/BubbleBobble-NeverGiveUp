import torch

env_name = 'BubbleBobble-Nes'
LEVEL_SET = ['Level01','Level02','Level03','Level04','Level05','Level06','Level07','Level08','Level09','Level10',
             'Level11','Level12','Level13','Level14','Level15','Level16','Level17','Level18','Level19','Level20',
             'Level21','Level22','Level23','Level24','Level25','Level26','Level27','Level28','Level29','Level30',
             'Level31','Level32','Level33','Level34','Level35','Level36','Level37','Level38','Level39','Level40',
             'Level41','Level42','Level43','Level44','Level45','Level46','Level47','Level48','Level49','Level50',
             'Level51','Level52','Level53','Level54','Level55','Level56','Level57','Level58','Level59','Level60',
             'Level61','Level62','Level63','Level64','Level65','Level66','Level67','Level68','Level69','Level70',
             'Level71','Level72','Level73','Level74','Level75','Level76','Level77','Level78','Level79','Level80',
             'Level81','Level82','Level83','Level84','Level85','Level86','Level87','Level88','Level89','Level90',
             'Level91','Level92','Level93','Level94','Level95','Level96','Level97','Level98','Level99']


gamma = 0.99
lr = 0.001
goal_score = 200000
log_interval = 10
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

lambda_gae = 0.96
epsilon_clip = 0.2
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
batch_size = 8
epoch_k = 10