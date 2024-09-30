

import argparse
import data as Dataset
from config import Config
from util.logging import init_logging, make_logging_dir
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import init_dist
from util.distributed import master_only_print as print
import time
if 0:
    import wandb
    wandb.init(project="PiRender", sync_tensorboard=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default= './config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--person_number', type=int, default=-1)
    parser.add_argument('--device', type=int, default=-2)
    args = parser.parse_args()
    return args



import os 

#@profile
#def main():
if __name__ == '__main__':
    # get training options
    tmp = None
    args = parse_args()

    if args.device != -2:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=True)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = opt.local_rank
    else:
        opt.device = "cuda:0"
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)
    # create a dataset
    
    if args.person_number != -1:
        opt.data.person_number = args.person_number
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)
    #1/0
   
    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset)
    opt_logdir = opt.logdir
    opt.logdir = opt.trainer.recovery_path # "result/face/"#TODO: fix
    if opt.data.decapirender_ckpt_format:
        current_epoch, current_iteration = trainer.load_checkpoint(opt, None, del_map = False)##args.which_iter)   
    else:
        current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter, del_map = False)
    
    #Chkp TODO: fix
    opt.logdir = opt_logdir
    # training flag
    max_epoch = opt.max_epoch
    #1/0
    if args.debug:
        tmp_dat = iter(train_dataset)
        data = next(tmp_dat)
        trainer.test_everything(train_dataset, val_dataset, current_epoch, current_iteration)
        exit()
    # Start training.
    
    time_limit = 3600 * 2
    start_time = time.time()
    for epoch in range(current_epoch, opt.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_dataset.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_dataset):
            data = trainer.start_of_iteration(data, current_iteration)
            trainer.optimize_parameters(data)
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)
 
            if current_iteration >= opt.max_iter:
                print('Done with training!!!')
                break
        current_epoch += 1
        trainer.end_of_epoch(data, val_dataset, current_epoch, current_iteration)
        if time_limit != -1 and time.time() - start_time > time_limit:
            print('Time limit')
            break

#if __name__ == '__main__':
#main()
