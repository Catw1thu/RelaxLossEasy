#!/usr/bin/env python
# coding: utf-8


import torch
import my_models as models
import my_utils as ut
import attacks as at
from options import args_parser
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from time import ctime
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


args = args_parser()


# datasets & dataloaders 
tr_ds, te_ds = ut.get_datasets(args.data, augment=args.data_augment)
v_size = int(len(tr_ds)*0.1)
tr_ds, v_ds = torch.utils.data.random_split(tr_ds, [len(tr_ds)-v_size, v_size])
tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
v_loader = DataLoader(v_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
te_loader = DataLoader(te_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)


# model & loss & etc.
# 在 main.py 中, model.to(args.device) 之前
print(f"Value of args.device: {args.device}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

model = models.resnet20().to(args.device)
criterion = torch.nn.CrossEntropyLoss()
#opt = optim.Adam(model.parameters())
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, verbose=True)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#adjusted for 100 epochs
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 70, 90], gamma=0.1, verbose=False) 
es = ut.EarlyStopping(patience=10, verbose=True)

# loggers
file_name = f"""time:{ctime()}_data:{args.data}_epoch:{args.num_epochs}_earlyStop:{args.early_stop}_dataAugment:{args.data_augment}_relaxAlpha:{args.relax_alpha}"""
writer = SummaryWriter('../logs/' + file_name)


for ep in tqdm(range(args.num_epochs)):
    model.train()
    tr_loss, tr_acc = 0, 0
    for _, (inp, lbl) in enumerate(tr_loader):
        inp, lbl = inp.to(device=args.device, non_blocking=True), \
            lbl.to(device=args.device, non_blocking=True)
        out = model(inp)
        loss = criterion(out, lbl)
        opt.zero_grad()
        
        if loss >= args.relax_alpha:
            loss.backward()
        else:
            if ep % 2 == 0:
                # just negate the loss to do grad ascent
                relax_loss = -loss
            else:
                # flatten probs, compute loss wrt soft labels
                with torch.no_grad():
                    prob_gt = F.softmax(out, dim=1)[torch.arange(lbl.size(0)), lbl]
                    prob_ngt = (1.0 - prob_gt) / (args.num_classes - 1)
                    onehot = F.one_hot(lbl, num_classes=10)
                    soft_labels = onehot * prob_gt.unsqueeze(-1).repeat(1, args.num_classes)\
                        + (1 - onehot) * prob_ngt.unsqueeze(-1).repeat(1, args.num_classes)
                relax_loss = criterion(out, soft_labels)    
            relax_loss.backward()
                
        opt.step()
        # keeping track of tr loss
        tr_loss += loss.item()*out.shape[0]
        _, pred_lbl = torch.max(out, 1)
        tr_acc += torch.sum(torch.eq(pred_lbl.view(-1), lbl)).item()
      
    # inference on val data after epoch
    with torch.inference_mode():
        tr_loss, tr_acc = tr_loss/len(tr_loader.dataset), tr_acc/len(tr_loader.dataset)
        v_loss, v_acc = ut.get_loss_n_accuracy(model, v_loader, device=args.device)
        #loggers
        writer.add_scalar('Train/Loss', tr_loss, ep)
        writer.add_scalar('Train/Acc', tr_acc, ep)
        writer.add_scalar('Val/Loss', v_loss, ep)
        writer.add_scalar('Val/Acc', v_acc, ep)
        print(f'|Tr/Val Loss: {tr_loss:.3f} / {v_loss:.3f}|')
        print(f'|Tr/Val Acc: {tr_acc:.3f} / {v_acc:.3f}|')
        # lr scheduling
        scheduler.step()
        # early-stopping 
        if args.early_stop:
            es(v_loss)
            if es.early_stop:
                print("Early stopping")
                break    


# inference of test data
te_loss, te_acc = ut.get_loss_n_accuracy(model, te_loader, device=args.device)
writer.add_scalar('Test/Loss', te_loss, 0)
writer.add_scalar('Test/Acc', te_acc, 0)
print(f'|Te Loss/Acc: {te_loss:.3f} / {te_acc:.3f}|')


# loss vals for all data pts in tr/test
tr_losses = ut.get_loss_vals(model, tr_loader, device=args.device)
te_losses = ut.get_loss_vals(model, te_loader, device=args.device)


# emp. var/mean of losses
tr_loss_var, tr_loss_mean = torch.var_mean(tr_losses, unbiased=False)
te_loss_var, te_loss_mean = torch.var_mean(te_losses, unbiased=False)
writer.add_scalar('Train/Avg.Loss', tr_loss_mean, 0)
writer.add_scalar('Train/Sample Var.', tr_loss_var, 0)
writer.add_scalar('Test/Avg.Loss', te_loss_mean, 0)
writer.add_scalar('Test/Sample Var.', te_loss_var, 0)
print(f'|Tr Avg.Loss/Sample Var.: {tr_loss_mean:.3f} / {tr_loss_var:.3f}|')
print(f'|Te Avg.Loss/Sample Var.: {te_loss_mean:.3f} / {te_loss_var:.3f}|')


# performance of attack
# attack performance (bal.acc, true-positive-rate, false-positive-rate) 
(ds_bacc, ds_tpr, ds_fpr), _ = at.mia_by_threshold(model, tr_loader, te_loader, threshold=tr_loss_mean, device=args.device)
writer.add_scalar('Attack/Bacc', ds_bacc, 0)
writer.add_scalar('Attack/TPR', ds_tpr, 0)
writer.add_scalar('Attack/FPR', ds_fpr, 0)



# plot loss histograms on [0-5] range
# based on https://stackoverflow.com/questions/47999159/normalizing-two-histograms-in-the-same-plot
num_bin = 50
bin_lims = np.linspace(0,5,num_bin+1)
bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
bin_widths = bin_lims[1:]-bin_lims[:-1]

##computing the histograms
hist1, _ = np.histogram(tr_losses.cpu().detach().numpy(), bins=bin_lims)
hist2, _ = np.histogram(te_losses.cpu().detach().numpy(), bins=bin_lims)

##normalizing
hist1b = hist1/np.sum(hist1)
hist2b = hist2/np.sum(hist2)

# plotting
plt.bar(bin_centers, hist1b, width = bin_widths, alpha = 0.5, align = 'center', label='train', color='tab:green')
plt.bar(bin_centers, hist2b, width = bin_widths, alpha = 0.5, align = 'center', label='test', color='tab:red')
plt.legend(loc='upper right')
plt.ylabel('Normalized Frequency')
plt.xlabel('Loss')
plt.xticks(np.arange(0, 5, 0.5))
plt.tight_layout()
plt.savefig(f'../plots/{file_name}', dpi=150, format='png')

# fin
writer.flush()
writer.close()    