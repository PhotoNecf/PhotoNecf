from torch.utils.data import DataLoader
import torch
from utils.datasetnp import DatasetDWT
from models.network_dncnn import DnCNN, FDnCNN
from models.network_ffdnet import FFDNet
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from utils.loss import HardTripletLoss
import argparse
from utils.post_process import extract_prnu_single
from loguru import logger
import os


def train(opt):
    dirs = os.path.join('ckps', opt.save_path)
    if not os.path.exists(dirs): os.makedirs(dirs)
    print(dirs, opt.batch_size)

    # To speed up the reading process, we convert the training images to npy format in advance
    # by function `convert_train_image` at utils/pre_process.py
    train_set = DatasetDWT(file_csv=opt.data_dir, file_root='path/to/train/data/with/npy/format')
    train_loader = DataLoader(train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              drop_last=True,
                              pin_memory=True)

    # To speed up the reading process, we convert the training images to npy format in advance
    # by function `convert_test_image` at utils/pre_process.py
    test_set = DatasetDWT(file_csv=opt.test_dir, file_root='path/to/test/data/with/npy/format')
    test_loader = DataLoader(test_set,
                             batch_size=15,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             drop_last=False,
                             pin_memory=True)

    model = FFDNet(in_nc=opt.image_channel,
                   out_nc=opt.image_channel,
                   nc=opt.net_channel,
                   nb=opt.layer_num,
                   act_mode=opt.act_mode)

    if opt.pretrained_model:
        print('load pretrained...')
        try:
            state_dict = torch.load(opt.pretrained_model)
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.model', 'model')
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=True)
        except Exception as e:
            print("unmatched pretrained model params with Exception" + str(e))

    if opt.device != "cpu":
        model = torch.nn.DataParallel(model.cuda())

    criterion = HardTripletLoss(margin=0.2, hardest=True).to(opt.device)
    criterion_l2 = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    schedulers = []
    schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                                               opt.scheduler_milestones,
                                               opt.scheduler_gamma
                                               ))

    current_step = 0

    model.train()
    print("start training...")
    for epoch in range(opt.max_epoches):
        size = len(train_set)
        training_loss = 0
        for i, (image, prnu, label, expo, iso) in enumerate(train_loader):
            image, prnu, label, expo, iso = image.to(opt.device), prnu.to(opt.device), label.to(opt.device), expo.to(
                opt.device), iso.to(opt.device)
            sigma = iso.view(-1, 1, 1, 1)
            embedding = model(image, sigma)
            loss_l2 = criterion_l2(embedding, prnu)
            loss_triplet = criterion(embedding, label)
            loss_reg = torch.mean(torch.linalg.norm(torch.flatten(embedding, start_dim=1), dim=1))
            optimizer.zero_grad()
            loss = loss_l2 + 0.01 * (loss_triplet + loss_reg * opt.loss_ratio)
            loss.backward()
            optimizer.step()

            current_step += 1
            for scheduler in schedulers:
                scheduler.step(current_step)

            if i % 1 == 0:
                loss, current = loss.item(), i * len(image)
                training_loss += loss
                print(f"[epoch {epoch:>5d}] total_loss: {loss:>7f}  l2_loss: {loss_l2:>7f} \
                triplet_loss: {loss_triplet:>7f} regulation_loss: {loss_reg:>7f} [{current:>5d}/{size:>5d}]")
        train_loss = training_loss / size * opt.batch_size

        if epoch % opt.test_freq == 0:
            embeds, labels = [], []
            for i, (image, prnu, label, expo, iso) in enumerate(test_loader):
                image, expo, iso = image.to(opt.device), expo.to(opt.device), iso.to(opt.device)
                with torch.no_grad():
                    sigma = iso.view(-1, 1, 1, 1)
                    embedding = model(image, sigma)
                    embeds.append(embedding.cpu())
                    labels.append(label.cpu())

            from utils import evaluate
            fp_array = torch.cat(embeds, dim=0).squeeze().detach().numpy()
            cc_aligned_rot = evaluate.aligned_cc_torch(fp_array, fp_array)['ncc']
            device_list = torch.cat(labels, dim=0).numpy()
            gt_array = evaluate.gt(device_list, device_list)
            stats_result = evaluate.stats(cc_aligned_rot, gt_array)

            prnu_array = extract_prnu_single(fp_array)
            cc_aligned_rot = evaluate.aligned_cc_torch(prnu_array, prnu_array)['ncc']
            device_list = torch.cat(labels, dim=0).numpy()
            gt_array = evaluate.gt(device_list, device_list)
            stats_result_prnu = evaluate.stats(cc_aligned_rot, gt_array)
            print_result2 = f"test ori auc: {stats_result['auc']}, eer {stats_result['eer']}. prnu auc: {stats_result_prnu['auc']}, eer {stats_result_prnu['eer']}"
            logger.info(print_result2)
        print_result = "finish epoch " + str(epoch) + ": train loss:" + str(train_loss)

        logger.info(print_result)

        with open(os.path.join(dirs, 'log.txt'), "a") as myfile:
            myfile.write(print_result + "\n")
            if epoch % opt.test_freq == 0:
                myfile.write(print_result2 + "\n")

        if epoch % 1 == 0:
            model_path = os.path.join(dirs, str(epoch) + "_model.pth")
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./utils/train_sample_id.csv', help='training data path')
    parser.add_argument('--test_dir', type=str, default='./utils/test_aut_nopaper_noburst_each20_1.csv',
                        help='test_data')

    parser.add_argument('--test-freq', type=int, default=10, help='test frequency')
    parser.add_argument('--batch-size', type=int, default=2048, help='total batch size for all GPUs')
    parser.add_argument('--num-workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--image-channel', type=int, default=1, help='[1 or 3] for image channels')
    parser.add_argument('--net-channel', type=int, default=64, help='net channels for DNCNN networks')
    parser.add_argument('--layer-num', type=int, default=15, help='layer numbers for DNCNN networks')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate for all GPUs')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay for all GPUs')
    parser.add_argument('--loss-ratio', type=float, default=1e-3, help='loss ratio between regulation term and triplet')
    parser.add_argument('--act-mode', type=str, default='R', help='BR for BN+ReLU | R for ReLU')
    parser.add_argument('--pretrained-model', type=str, default='./ckps/ffdnet_pretrained/ffdnet_gray.pth',
                        help='DNCNN pretrianed model path')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--scheduler-milestones', default=[200000, 400000, 600000, 800000, 1000000, 2000000],
                        help='MultiStepLR milestones')
    parser.add_argument('--scheduler-gamma', default=0.5, help='MultiStepLR gamma')
    parser.add_argument('--max-epoches', type=int, default=300)
    parser.add_argument('--save-path', type=str, default='savepath', help='model save path')

    opt = parser.parse_args()
    train(opt)
