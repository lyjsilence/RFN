import os
import time
import numpy as np
import pandas as pd
import torch
import utils
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR


def Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id, optim='adam', epoch_max=200):
    '''
    if args.model_name == 'RFN':
        param_groups = [
            {'params': model.mgn_RFN.parameters(), 'lr': args.lr},
            {'params': model.norm_func.parameters(), 'lr': args.lr},
            {'params': model.joint_RFN.parameters(), 'lr': args.lr}
        ]
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        '''
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=args.lr,
                                 weight_decay=args.weight_decay)

    scheduler = ExponentialLR(optimizer, gamma=0.99, verbose=False)

    min_nll = np.inf

    if args.model_name == 'RFN':
        save_path = os.path.join(args.save_dirs, args.model_name, args.marginal, args.type, 'exp_' + str(exp_id))
    else:
        save_path = os.path.join(args.save_dirs, args.model_name, args.type, 'exp_' + str(exp_id))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()

    for epoch in range(1, epoch_max + 1):
        '''Model training'''
        model.train()
        epoch_start_time = time.time()

        training_epoch_nll_loss = []
        training_epoch_reg_loss = []
        training_epoch_loss = []

        for i, batch_ts in enumerate(dl_train):
            sample_idx = batch_ts["sample_idx"]
            obs_times = batch_ts["obs_times"]
            event_pt = batch_ts["event_pt"]
            X = batch_ts["X"].to(device)
            M = batch_ts["M"].to(device)
            batch_idx = batch_ts["batch_idx"]

            loss_nll, loss_reg \
                = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=args.dt)

            loss = loss_nll + loss_reg

            training_epoch_nll_loss.append(loss_nll.item())
            training_epoch_reg_loss.append(loss_reg.item())
            training_epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 20, norm_type=2)
            optimizer.step()

            print('\rEpoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, NLL Loss: {:.4f}, Reg Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, epoch_max, i + 1, len(dl_train), np.sum(training_epoch_loss) / len(training_epoch_loss),
                          np.sum(training_epoch_nll_loss) / len(training_epoch_nll_loss),
                          np.sum(training_epoch_reg_loss) / len(training_epoch_reg_loss),
                          time.time() - epoch_start_time), end='')

        if args.log:
            df_log_val.loc[epoch, 'Epoch'] = epoch
            df_log_val.loc[epoch, 'Time elapsed'] = time.time() - epoch_start_time
            df_log_val.loc[epoch, 'Loss_NLL_train'] = np.sum(training_epoch_nll_loss) / len(training_epoch_nll_loss)

        '''Model validation'''
        if epoch >= args.val_start:
            with torch.no_grad():
                model.eval()

                val_epoch_nll_loss = []

                for i, batch_ts_val in enumerate(dl_val):

                    sample_idx = batch_ts_val["sample_idx"]
                    obs_times = batch_ts_val["obs_times"]
                    event_pt = batch_ts_val["event_pt"]
                    X = batch_ts_val["X"].to(device)
                    M = batch_ts_val["M"].to(device)
                    batch_idx = batch_ts_val["batch_idx"]

                    # run the model for test
                    val_nll, val_reg \
                        = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=args.dt)

                    val_epoch_nll_loss.append(val_nll.item())

                # save the best model parameter
                if val_nll < min_nll:
                    print('Saving...')
                    min_nll = val_nll
                    model_save_path = os.path.join(save_path, args.model_name+'_sim.pkl')
                    torch.save(model.state_dict(), model_save_path)

                print(f"Validation: "
                      f"NLL={np.sum(val_epoch_nll_loss)/len(val_epoch_nll_loss):.4f}")

                if args.log:
                    df_log_val.loc[epoch, 'Loss_NLL_val'] = np.sum(val_epoch_nll_loss)/len(val_epoch_nll_loss)

            if args.viz:
                for i, batch_ts_test in enumerate(dl_test):
                    sample_idx = batch_ts_test["sample_idx"]
                    obs_times = batch_ts_test["obs_times"]
                    event_pt = batch_ts_test["event_pt"]
                    X = batch_ts_test["X"].to(device)
                    M = batch_ts_test["M"].to(device)
                    batch_idx = batch_ts_test["batch_idx"]
                    t_corr = [0.3, 0.6, 0.9]
                    _, _, _, _ = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device,
                                          dt=args.dt, viz=True, val=True, t_corr=t_corr)

                    viz_save_path = os.path.join(save_path, 'figs')
                    if not os.path.exists(viz_save_path):
                        os.makedirs(viz_save_path)
                    plt.savefig(os.path.join(viz_save_path, '{:03d}.jpg'.format(epoch)))

        else:
            print('')

        scheduler.step(epoch)
        torch.cuda.empty_cache()

    '''Model test'''
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))

        test_epoch_nll_loss = []
        test_crps = []
        test_crps_sum = []
        test_ND = []

        for i, batch_ts_test in enumerate(dl_test):

            sample_idx = batch_ts_test["sample_idx"]
            obs_times = batch_ts_test["obs_times"]
            event_pt = batch_ts_test["event_pt"]
            X = batch_ts_test["X"].to(device)
            M = batch_ts_test["M"].to(device)
            batch_idx = batch_ts_test["batch_idx"]

            # run the model for test
            test_nll, crps, crps_sum, ND \
                = model(obs_times, event_pt, sample_idx, X, M, batch_idx, device, dt=args.dt, val=True)

            test_epoch_nll_loss.append(test_nll.item())
            test_crps.append(crps)
            test_crps_sum.append(crps_sum)
            test_ND.append(ND)

        # save the best model parameter
        print(f"Test: "
              f"NLL={np.sum(test_epoch_nll_loss)/len(test_epoch_nll_loss):.4f}, "
              f"CRPS={np.sum(test_crps)/len(test_crps):.4f},"
              f"CRPS_sum={np.sum(test_crps_sum)/len(test_crps_sum):.4f},"
              f"ND={np.sum(test_ND)/len(test_ND):.4f}")

        torch.cuda.empty_cache()

    if args.log:
        df_log_test.loc[0, 'Loss_NLL_test'] = np.sum(test_epoch_nll_loss)/len(test_epoch_nll_loss)
        df_log_test.loc[0, 'CRPS_test'] = np.sum(test_crps)/len(test_crps)
        df_log_test.loc[0, 'CRPS_sum_test'] = np.sum(test_crps_sum)/len(test_crps_sum)
        df_log_test.loc[0, 'ND_test'] = np.sum(test_ND)/len(test_ND)

    if args.log:
        df_log_val.to_csv(os.path.join(save_path, 'training_log.csv'))
        df_log_test.to_csv(os.path.join(save_path, 'test_log.csv'))