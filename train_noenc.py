# An SNN version attempt of YOLOX
# Original codes come from https://github.com/bubbliiiing/yolox-pytorch

import torch
import torch.nn as nn
import torch, torchvision
import argparse
import datetime
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from nets.yolo import YoloBody
from nets.yolo_training import ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config, get_lr
from utils.utils_fit import fit_one_epoch

from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, layer
# from spikingjelly.activation_based.encoding import PoissonEncoder

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

if __name__ == '__main__':      #主函数
    # 训练初始化
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda            = True
    # 随机数种子固定
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/coco_classes.txt'
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolox_s.pth'      # 'models_data/yolox_s.pth' model_data/best_epoch_weights.pth
    #------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    #   所使用的YoloX的版本。nano、tiny、s、m、l、x
    #------------------------------------------------------#
    phi             = 's'
    #------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    #
    #   余弦退火算法的参数放到下面的lr_decay_type中设置
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.5
 #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 250
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。 从头开始训练设置为False
    #------------------------------------------------------------------#
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.001
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.001
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #   学习率设置代码有大问题
    #------------------------------------------------------------------#
    optimizer_type      = "adam"        # SNN训练先用Adam
    momentum            = 0.9
    weight_decay        = 0          # Adam的weight decay就是0，SGD看情况
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos、multistep
    #------------------------------------------------------------------#
    lr_decay_type       = "step"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 50
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 4

    #----------------------------------------------------#
    #   获得图片路径和标签m_classes
    #----------------------------------------------------#
    train_annotation_path   = 'coco_train.txt'  # '2007_train.txt'  VOC用  'coco_train.txt'  coco用
    val_annotation_path     = 'coco_val.txt'    # '2007_val.txt'    'coco_val.txt' 

    #------------------------------------------------------#
    #   设置用到的显卡，Windows下不使用DDP，仅使用DP
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cudnn.benchmark = True   # cudnn auto-tuner
    else:
        device = torch.device("cpu")
    local_rank      = 0
    rank            = 0
        
    #----------------------------------------------------#
    #   获取classes和anchor，类别数量VOC/COCO对应80
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   创建yolo模型， 这里模型已经换成带SNN, 添加SNN编码器
    #------------------------------------------------------#
    model = YoloBody(num_classes, phi).to(device)

    weights_init(model)
    model = torch.nn.DataParallel(model)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        pretrained_dict = torch.load(model_path, map_location = lambda storage, loc: storage)
        # new_dict = {}
        # for k,v in pretrained_dict.items():
        #     if k.startswith("module."):
        #         k=k.replace("module.","")
        #     new_dict[k] = v
        # load_key, no_load_key, temp_dict = [], [], {}
        # for k, v in pretrained_dict.items():
        #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        #         temp_dict[k] = v
        #         load_key.append(k)
        #     else:
        #         no_load_key.append(k)
        # 更新并加载权重
        # 多卡训练加载单卡模型
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        # 这里看看怎么改能直接用前序模型训练好的权重，可能有一步要删除不匹配的权重字典
        model.load_state_dict(pretrained_dict, strict=False)

    #----------------------#
    #   获得损失函数，理论上损失函数应该不用动
    #----------------------#
    yolo_loss    = YOLOLoss(num_classes, fp16).to(device)
    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #   这里先使用混合精度训练看看，torch版本为1.9.1+cu111
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
  
    #----------------------------#
    #   权值平滑
    #----------------------------#
    # ema = ModelEMA(model)
    ema = None
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
   
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # SNN单多步训练，放在这里可以实现冻结部分权重训练
        model = layer.StepModeContainer(True, model)      # 单步用这个
        # model = layer.MultiStepContainer(model)         # 多步用这个
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率，这块计划大改
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个epoch长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                    
                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                   
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 这块是训练过程了，一次按照一个epoch训练，分类检测还好，其他任务不一定了
            # 设置成SNN模式
            functional.set_step_mode(model, step_mode='s')

            # 给训练的fit_one_epoch函数搬进来了
            loss        = 0
            val_loss    = 0
            
            if local_rank == 0:
                print('Start Train')
                pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)
            
            model.train()
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_step:
                    break
                
                images, targets = batch[0], batch[1]
                # with torch.no_grad():
                if Cuda:
                    images  = images.to(device)
                    targets = [ann.to(device) for ann in targets]
                
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                if not fp16:
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs         = model(images)

                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss_value = yolo_loss(outputs, targets)

                    #----------------------#
                    #   反向传播
                    #----------------------#
                    # torch.autograd.set_detect_anomaly(True)
                    loss_value.backward()
                    optimizer.step()
                    functional.reset_net(model)
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(images)
                        #----------------------#
                        #   计算损失
                        #----------------------#
                        loss_value = yolo_loss(outputs, targets)

                    #----------------------#
                    #   反向传播
                    #----------------------#
                    scaler.scale(loss_value).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    functional.reset_net(model)
                if ema:
                    ema.update(model)

                loss += loss_value.item()
                
                if local_rank == 0:
                    pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                        'lr'    : get_lr(optimizer)})
                    pbar.update(1)

            if local_rank == 0:
                pbar.close()
                print('Finish Train')
                print('Start Validation')
                pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)

            if ema:
                model_train_eval = ema.ema
            else:
                model_train_eval = model.eval()
                
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images  = images.cuda(local_rank)
                        targets = [ann.cuda(local_rank) for ann in targets]
                    #----------------------#
                    #   清零梯度
                    #----------------------#
                    optimizer.zero_grad()
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    # outputs         = model_train_eval(enc(images))
                    outputs         = model_train_eval(images)

                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss_value = yolo_loss(outputs, targets)

                    functional.reset_net(model_train_eval)

                val_loss += loss_value.item()
                if local_rank == 0:
                    pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                    pbar.update(1)


            if local_rank == 0:
                pbar.close()
                print('Finish Validation')
                loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
                eval_callback.on_epoch_end(epoch + 1, model_train_eval)
                print('Epoch:'+ str(epoch + 1) + '/' + str(UnFreeze_Epoch))
                print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
                
                #-----------------------------------------------#
                #   保存权值
                #-----------------------------------------------#
                if ema:
                    save_state_dict = ema.ema.state_dict()
                else:
                    save_state_dict = model.state_dict()

                if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
                    torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

                if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                    print('Save best model to best_epoch_weights.pth')
                    torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
                    
                torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

        if local_rank == 0:
            loss_history.writer.close()



    


