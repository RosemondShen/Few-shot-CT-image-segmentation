import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import few_shot_segmentor as fs
import attention_few_shot_segmentor as sn
from nn_common_modules import losses
import copy
from utils.log_utils import LogWriter
from shot_batch_sampler import get_lab_list, get_image_num_dict, one_shot_batch_sampler, get_image_and_masks, batch
import utils.evaluator as eu
from utils.common_utils import load_checkpoint, split_batch, create_if_not
from settings import Settings
import matplotlib.pyplot as plt


# labels = ["Liver00", "Liver01", "Lung00", "Pancreas00", "Pancreas01", "Spleen00", "Colon00"]

def train(common_params, net_params, attention, refinement):

    model_prefix = 'base_model_'
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7']
    for fold in folds:

        # exp_name = "base_model_fold1"
        exp_name = model_prefix + fold
        # final_model_path = "saved_models/base_model_fold1.pth.tar"
        final_model_path = os.path.join("/home/yzhang14/Data/MSD/saved_models", exp_name + '.pth.tar')

        # few_shot_model = fs.FewShotSegmentorDoubleSDnet(net_params)
        few_shot_model = sn.FewShotSegmentorDoubleSDnet(net_params)
        device = common_params["device"]
        best_ds_mean = 0
        best_ds_mean_epoch = 1
        epoch_num = 20

        loss_func = losses.DiceLoss()
        exp_dir = '/home/yzhang14/Data/MSD/experiments'
        log_dir = '/home/yzhang14/Data/MSD/logs'
        use_last_checkpoint = True
        exp_dir_path = os.path.join(exp_dir, exp_name)
        create_if_not(exp_dir_path)
        create_if_not(os.path.join(exp_dir_path, 'checkpoints'))
        logwriter = LogWriter(1, log_dir, exp_name, use_last_checkpoint)
        logwriter.log('The processing fold is: %s' % fold)

        if torch.cuda.is_available():
            loss_func = loss_func.cuda(device)
        else:
            loss_func = loss_func

        optim_c = torch.optim.SGD(few_shot_model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.0001)
        # optim_s = optim(few_shot_model.segmentor.parameters(), lr=1e-9, momentum=0.99, weight_decay=0.0001)
        scheduler_c = lr_scheduler.StepLR(optim_c, step_size=10, gamma=0.1)
        # scheduler_s = lr_scheduler.StepLR(optim_s, step_size=10, gamma=0.1)

        if use_last_checkpoint:
            load_checkpoint(logwriter, few_shot_model, optim_c, scheduler_c, exp_dir_path, epoch=None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            few_shot_model.cuda(device)

        for phase in ['train', 'val']:

            lab_list = get_lab_list(fold, phase)
            data_set = get_image_and_masks(lab_list, phase)

            for epoch in range(1, epoch_num+1):
                logwriter.log("\n========= Epoch [ %d  /  %d ] START =========" % (epoch, epoch_num))
                logwriter.log("\n<<<=== phase: %s ===>>>" % phase)
                loss_arr = []
                y_list = []
                out_list = []

                scheduler_c.step()
                # scheduler_s.step()

                if phase == 'train':   
                    few_shot_model.train()

                else:
                    few_shot_model.eval()

                train_loader = torch.utils.data.DataLoader(data_set, batch_size=4)
                val_loader = torch.utils.data.DataLoader(data_set, batch_size=4)

                data_loader = {'train': train_loader, 'val': val_loader}

                for i_batch, sampled_batch in enumerate(data_loader[phase]):

                    input1 = sampled_batch[0]
                    y1 = sampled_batch[1]

                    if few_shot_model.is_cuda:
                        input1, y1 = input1.cuda(device), y1.cuda(device)
                        
                    optim_c.zero_grad()  # clear gradients for this training step
                    output = few_shot_model(input1)

                    loss = loss_func(output, y1, binary=True)
                    loss.backward()  # backpropagation, compute gradients
                    if phase == 'train':
                        optim_c.step()  # apply gradients
                        # optim_s.step()  # apply gradients

                    print("loss.item", loss.item())

                    loss_arr.append(loss.item())
                    out_list.append(output.cpu())
                    y_list.append(y1.cpu())
                    print(len(out_list), len(y_list))

                    del input1, y1, output, loss# , input2, y2,
                    torch.cuda.empty_cache()

                if phase == "train":
                    logwriter.log('saving checkpoint ....')
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': few_shot_model.state_dict(),
                        'optimizer_c': optim_c.state_dict(),
                        'scheduler_c': scheduler_c.state_dict(),
                        'best_ds_mean_epoch': best_ds_mean_epoch},
                        os.path.join(exp_dir_path, 'checkpoints', 'checkpoint_epoch_' + str(epoch) + '.pth.tar'))

                with torch.no_grad():
                    y_arr = torch.cat(y_list)
                    out_arr = torch.cat(out_list)

                    ds_mean = logwriter.dice_score_per_epoch(out_arr, y_arr, epoch)
                    if phase == 'val':
                        if ds_mean > best_ds_mean:
                            best_ds_mean = ds_mean
                            best_ds_mean_epoch = epoch
                logwriter.log("==== Epoch [" + str(epoch) + " / " + str(1) + "] DONE ====")
        logwriter.log('FINISH.')
        # load_checkpoint(logwriter, few_shot_model, optim_c, optim_s, scheduler_c, scheduler_s, exp_dir_path, best_ds_mean_epoch)
        load_checkpoint(logwriter, few_shot_model, optim_c, scheduler_c, exp_dir_path, best_ds_mean_epoch)
        torch.save(few_shot_model, final_model_path)

        logwriter.log("final model saved at: " + str(final_model_path))
        logwriter.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True,
                        help='run mode, valid values are train and eval')
    parser.add_argument('--device', '-d', required=False,
                        help='device to run on')
    parser.add_argument('--attention', '-a', type=int, default=0,
                        help='comparison attention mechanism')
    parser.add_argument('--refinement', '-r', type=int, default=0,
                        help='organ boundary refinement')
    args = parser.parse_args()
    mode = args.mode
    attention = args.attention
    refinement = args.refinement

    settings = Settings()
    common_params, net_params, train_params, eval_params = settings['COMMON'], settings['NETWORK'], settings['TRAINING'], settings['EVAL']
    torch.set_default_tensor_type('torch.FloatTensor')
    labels = common_params["labels"]

    if args.device is not None:
        common_params['device'] = args.device

    if attention == 0:
        attention = False
    elif attention == 1:
        attention = True
    else:
        raise ValueError("only 0 or 1 allowed for comparison attention mechanism")

    if refinement == 0:
        refinement = False
    elif refinement == 1:
        refinement = True
    else:
        raise ValueError("only 0 or 1 allowed for organ boundary refinement")

    if mode == 'train':
        print("Project phase: ", mode)
        train(common_params, net_params, attention, refinement)
    elif mode == 'eval':
        print("Project phase: ", mode)
        # evaluate(eval_params, net_params, common_params, train_params, attention, refinement)
    else:
        raise ValueError(
            'Invalid value for mode. only support values are train and eval')
