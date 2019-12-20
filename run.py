import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import few_shot_segmentor as fs
from nn_common_modules import losses

from utils.log_utils import LogWriter
from utils.shot_batch_sampler import get_lab_list, get_image_num_dict, one_shot_batch_sampler, get_image_and_masks, batch
import utils.evaluator as eu
from utils.common_utils import load_checkpoint, split_batch, create_if_not
from settings import Settings

settings = Settings()
common_params, net_params, train_params, eval_params = settings['COMMON'], settings['NETWORK'], settings['TRAINING'], settings['EVAL']
torch.set_default_tensor_type('torch.FloatTensor')

labels = common_params["labels"]
# labels = ["Liver00", "Liver01", "Lung00", "Pancreas00", "Pancreas01", "Spleen00", "Colon00"]


def train(net_params, attention, refinement):

    model_prefix = 'base_model_'
    folds = ['fold1'
        , 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7']
    for fold in folds:

        # exp_name = "base_model_fold1"
        exp_name = model_prefix + fold
        # final_model_path = "saved_models/base_model_fold1.pth.tar"
        final_model_path = os.path.join("/home/yzhang14/Data/MSD/saved_models", exp_name + '.pth.tar')

        few_shot_model = fs.FewShotSegmentorDoubleSDnet(net_params)
        device = 0
        num_class = 1
        start_epoch = 1
        num_epochs = 1
        best_ds_mean = 0
        best_ds_mean_epoch = 1
        warm_up_epoch = 900
        val_old = 0
        change_model = False
        current_model = 'seg'

        optim = torch.optim.SGD
        loss_func = losses.DiceLoss()
        exp_dir = '/home/yzhang14/Data/MSD/experiments'
        log_dir = '/home/yzhang14/Data/MSD/logs'
        use_last_checkpoint = True
        exp_dir_path = os.path.join(exp_dir, exp_name)
        create_if_not(exp_dir_path)
        create_if_not(os.path.join(exp_dir_path, 'checkpoints'))
        logwriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint)

        if torch.cuda.is_available():
            loss_func = loss_func.cuda(device)
        else:
            loss_func = loss_func

        optim_c = optim(few_shot_model.conditioner.parameters(), lr=1e-9, momentum=0.99, weight_decay=0.0001)
        optim_s = optim(few_shot_model.segmentor.parameters(), lr=1e-9, momentum=0.99, weight_decay=0.0001)
        scheduler_s = lr_scheduler.StepLR(optim_s, step_size=10, gamma=0.1)
        scheduler_c = lr_scheduler.StepLR(optim_c, step_size=10, gamma=0.001)

        if use_last_checkpoint:
            load_checkpoint(logwriter, few_shot_model, optim_c, optim_s, scheduler_c, scheduler_s, exp_dir_path, epoch=None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            few_shot_model.cuda(device)
        '''
        lab_list = get_lab_list(phase='train', fold=fold)
        img_num_list = get_image_num_dict(lab_list, phase='train')
        p = [val for val in img_num_list.values()]
        prob = p / np.sum(p)
        img_num_list_val = get_image_num_dict(lab_list, phase='val')
        p_val = [val for val in img_num_list_val.values()]
        prob_val = p_val / np.sum(p_val)
        logwriter.log('The processing fold is: %s' % fold)
        logwriter.log('START TRAINING......')
        '''
        lab_list = get_lab_list(fold, phase)
        data_set = get_image_and_masks(lab_list, phase)

        for epoch in range(start_epoch, num_epochs + 1):
            logwriter.log("\n========= Epoch [ %d  /  %d ] START =========" % (epoch, num_epochs))

            for phase in ['train', 'val']:
                logwriter.log("\n<<<=== phase: %s ===>>>" % phase)
                loss_arr = []
                y_list = []
                out_list = []

                if phase == 'train':
                    few_shot_model.train()
                else:
                    few_shot_model.eval()

                for j, b in enumerate(batch(data_set, 2)):

                    if phase == "train" and j == min(600, len(data_set)//2):
                        break
                    elif phase == "val" and j == min(60, len(data_set)//2):
                        break

                    print("====iteration time:", j, "====")
                    input1 = b[0][0]
                    y1 = b[0][1]
                    input2 = b[1][0]
                    y2 = b[1][1]
                    '''
                    if phase == "train":
                        input1, input2, y1, y2 = one_shot_batch_sampler(phase=phase, lab_list=lab_list,
                                                                        img_num_list=img_num_list, prob=prob)
                    else:
                        input1, input2, y1, y2 = one_shot_batch_sampler(phase=phase, lab_list=lab_list,
                                                                        img_num_list=img_num_list_val, prob=prob_val)
                    '''
                    condition_input = torch.cat((input1, y1), dim=1)
                    query_input = input2

                    if few_shot_model.is_cuda:
                        condition_input, query_input, y2, y1 = condition_input.cuda(device), query_input.cuda(device), \
                                                               y2.cuda(device), y1.cuda(device)
                    weights = few_shot_model.conditioner(condition_input)
                    output = few_shot_model.segmentor(query_input, weights)
                    loss = loss_func(output, y2, binary=True)
                    optim_s.zero_grad()  # clear gradients for this training step
                    optim_c.zero_grad()  #
                    loss.backward()  # backpropagation, compute gradients
                    if phase == 'train':
                        optim_s.step()  # apply gradients
                        optim_c.step()

                        scheduler_c.step()
                        scheduler_s.step()

                    print("loss.item", loss.item())

                    loss_arr.append(loss.item())
                    out_list.append(output.cpu())
                    y_list.append(y2.cpu())
                    print(len(out_list), len(y_list))

                    del input1, input2, y1, y2, output, loss
                    torch.cuda.empty_cache()

                if phase == "train":
                    logwriter.log('saving checkpoint ....')
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': few_shot_model.state_dict(),
                        'optimizer_c': optim_c.state_dict(),
                        'scheduler_c': scheduler_c.state_dict(),
                        'optimizer_s': optim_s.state_dict(),
                        'scheduler_s': scheduler_s.state_dict(),
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
            logwriter.log("==== Epoch [" + str(epoch) + " / " + str(num_epochs) + "] DONE ====")
        logwriter.log('FINISH.')
        logwriter.log('Saving model... %s' % final_model_path)
        load_checkpoint(logwriter, few_shot_model, optim_c, optim_s, scheduler_c, scheduler_s, exp_dir_path, best_ds_mean_epoch)
        torch.save(few_shot_model, final_model_path)

        logwriter.log("final model saved at: " + str(final_model_path))
        logwriter.close()


def evaluate(eval_params, net_params, common_params, train_params, attention, refinement):

    log_dir = "/home/yzhang14/Data/MSD/logs"
    device = 0
    exp_dir = "/home/yzhang14/Data/MSD/experiments"
    exp_name = common_params['exp_name']
    num_classes = 1
    num_epochs = 100
    save_predictions_dir = eval_params['save_predictions_dir']

    logwriter = LogWriter(num_classes, log_dir, exp_name)
    model_prefix = 'base_model_'
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7']

    for fold in folds:
        lab_list = get_lab_list(phase="test", fold=fold)
        # lab_list = ["Liver00", "Liver01", "Lung00", "Pancreas00", "Pancreas01", "Spleen00"]
        img_num_list = get_image_num_dict(lab_list, phase='test')
        p = [val for val in img_num_list.values()]
        prob = p / np.sum(p)
        print("The probability of choosing different labels are:", prob)
        query_label = lab_list[0]

        input1, input2, y1, y2 = one_shot_batch_sampler(phase='test', lab_list=lab_list, img_num_list=img_num_list,
                                                        prob=prob, batch_size=2)
        prediction_path = os.path.join(exp_dir, exp_name)
        prediction_path = prediction_path + "_" + fold
        prediction_path = os.path.join(prediction_path, save_predictions_dir)

        eval_model_path = os.path.join(common_params['save_model_dir'], model_prefix + fold + '.pth.tar')
        query_labels = get_lab_list('val', fold)

        avg_dice_score = eu.evaluate_dice_score(input1, input2, y1, y2, eval_model_path, prediction_path, num_epochs,
                                                query_label, device, fold=fold)
        logwriter.log(avg_dice_score)
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
    device = args.device
    attention = args.attention
    refinement = args.refinement

    settings = Settings()
    common_params, net_params, train_params, eval_params = settings['COMMON'], settings['NETWORK'], settings['TRAINING'], settings['EVAL']

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
        train(net_params, attention, refinement)
    elif mode == 'eval':
        print("Project phase: ", mode)
        evaluate(eval_params, net_params, common_params, train_params, attention, refinement)
    else:
        raise ValueError(
            'Invalid value for mode. only support values are train and eval')
