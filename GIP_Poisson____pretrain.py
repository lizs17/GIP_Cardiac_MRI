import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
from utils import reverse_order_np, fft2c_norm_np, ifft2c_norm_np, rss_coil_np
from utils import fft2c_norm_torch, ifft2c_norm_torch, R2C_cat_torch
from utils import C2R_insert_torch, C2R_insert_torch, metrics2D_np
from utils import save_dict_h5, read_dict_h5
from model import generator_G, generator_GCN
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--dataName", type=str, default='fs_0032_3T_slc1_p3')
parser.add_argument("--R", type=float, default=8.0)
parser.add_argument("--ACS1", type=int, default=10)
parser.add_argument("--ACS2", type=int, default=10)
args = parser.parse_args()
gpu = args.gpu
dataName = args.dataName
R = args.R
ACS1 = args.ACS1
ACS2 = args.ACS2
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
dtype_float_np = np.float32
dtype_complex_np = np.complex64
dtype_float_torch = torch.float32
dtype_complex_torch = torch.complex64
# ------
seed = 1
device = torch.device("cuda", gpu)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- parameters
# ----------------------------------------------------------------------------------------------------------------------
CH = 24
K = 7
# ------ G
lr_G = 5e-4
betas_G = (0.5, 0.98)
Niter_G = 10000
eval_every_G = 100
save_every_G = 5000
# ------ C
lr_C = 5e-4
betas_C = (0.5, 0.98)
Niter_C = 5000
eval_every_C = 100
save_every_C = 2500
# ------ tune
lr_tune = 5e-5
betas_tune = (0.5, 0.98)
Niter_tune = 15000
Niter_graph = 10000
eval_every_tune = 100
save_every_tune = 5000
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
postfix = ('_R' + str(R)) + ('_ACS' + str(ACS1) + 'x' + str(ACS2))
reconFolder = 'GIP_Poisson' + postfix + '/'
# ------
if not os.path.exists(reconFolder):
    os.mkdir(reconFolder)
caseFolder = os.path.join(reconFolder, dataName + '/')
if not os.path.exists(caseFolder):
    os.mkdir(caseFolder)
G_Folder = os.path.join(caseFolder, 'G/')
if not os.path.exists(G_Folder):
    os.mkdir(G_Folder)
C_Folder = os.path.join(caseFolder, 'C/')
if not os.path.exists(C_Folder):
    os.mkdir(C_Folder)
tune_Folder = os.path.join(caseFolder, 'tune/')
if not os.path.exists(tune_Folder):
    os.mkdir(tune_Folder)
log_G_Path = os.path.join(G_Folder, 'log_G.txt')
log_C_Path = os.path.join(C_Folder, 'log_C.txt')
log_tune_Path = os.path.join(tune_Folder, 'log_tune.txt')
# ------
nn_idx_path = os.path.join(tune_Folder, 'nn_idx.h5')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
dataFolder = 'data/'
maskFolder = 'mask/'
# ------
data = read_dict_h5(os.path.join(dataFolder, dataName + '.h5'))
# ------
imgGT = data['img']  # (Nx, Ny, Nt)
smaps = data['smap']  # (Nx, Ny, Nc)
# ------
Nx, Ny, Nt = imgGT.shape
Nc = smaps.shape[2]
imSize = (Ny, Nx)
# ------
maskName = 'maskPoisson' \
           + ('_Nx' + str(Nx) + '_Ny' + str(Ny) + '_Nt' + str(Nt)) \
           + ('_R' + str(R)) + ('_ACS' + str(ACS1) + 'x' + str(ACS2))
mask = read_dict_h5(os.path.join(maskFolder, maskName + '.h5'))['mask']  # (Nx, Ny, Nt)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------ dtype
imgGT = imgGT.astype(dtype_complex_np)
S_np = smaps.astype(dtype_complex_np)
M_np = mask.astype(dtype_float_np)
# ------ shape: (Nc, Nt, ch, Ny, Nx)
imgGT = np.reshape(reverse_order_np(imgGT), (1, Nt, 1, Ny, Nx))
S_np = np.reshape(reverse_order_np(S_np), (Nc, 1, 1, Ny, Nx))
M_np = np.reshape(reverse_order_np(M_np), (1, Nt, 1, Ny, Nx))
# ------ normalization
imgGT = imgGT / np.max(np.abs(imgGT))
S_np = S_np / rss_coil_np(S_np, dim=0)
# ------ undersampling
kdata = M_np * fft2c_norm_np(S_np * imgGT)  # (Nc, Nt, 1, Ny, Nx)
# ------ zero-filling recon
AHy_np = np.sum(np.conj(S_np) * ifft2c_norm_np(kdata), axis=0, keepdims=True)  # (1, Nt, 1, Ny, Nx)
# ------ tensor data
M_t = torch.tensor(M_np, dtype=dtype_float_torch).to(device)
S_t = torch.tensor(S_np, dtype=dtype_complex_torch).to(device)
y_t = torch.tensor(kdata, dtype=dtype_complex_torch).to(device)
# ------ ground-truth
imgRef_np = np.abs(imgGT)
imgRef_np = imgRef_np / np.max(np.abs(imgRef_np))
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- latent variable
# ----------------------------------------------------------------------------------------------------------------------
z_Path = os.path.join(caseFolder, 'z.h5')
if not os.path.exists(z_Path):
    z_np = np.random.normal(size=(1, 8, 8, 8))
    z_np = z_np.astype(dtype_float_np)
    save_dict_h5(h5_path=z_Path, np_dict={'z': z_np})
else:
    z_np = read_dict_h5(z_Path)['z']
# ------
z_t = torch.tensor(z_np, dtype=dtype_float_torch).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- GIP generator
# ----------------------------------------------------------------------------------------------------------------------
G = generator_G(device=device, imSize=imSize, Nt=Nt, ch=CH, ch_z=z_t.size(1), ch_out=2)
G.to(device)
G.weight_init()
# ------
C = generator_GCN(device=device, imSize=imSize, Nt=Nt, K=K, CH=CH, ch_out=2).to(device)
C.to(device)
C.weight_init()
# ======================================================================================================================
# ====================================================================================================================== pretrain G
# ======================================================================================================================
# ------ normalize k-space data
G.eval()
with torch.no_grad():
    _, xtmp = G(z_t)  # (Nt, 2) + imSize
    xtmp = R2C_cat_torch(xtmp, axis=1)  # (Nt, 1) + imSize
    xtmp = torch.reshape(xtmp, (1, Nt, 1) + imSize)  # (1, Nt, Nv) + imSize
    scaling_factor_G = np.linalg.norm(np.mean(xtmp.detach().cpu().numpy(),axis=1)) / np.linalg.norm(np.mean(AHy_np,axis=1))
y_G = y_t * scaling_factor_G
# save scaling factor
save_dict_h5(h5_path=os.path.join(G_Folder,'scaling_factor_G.h5'), np_dict={'scaling_factor_G': float(scaling_factor_G)})
# ------ pretrain G
# loss
lossFunc_G = nn.MSELoss(reduction='sum')
# optimizer
optimizer_G = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_G}
], betas=betas_G)
# train
tic_iter = time.time()
for iter in range(Niter_G):
    G.train()
    optimizer_G.zero_grad()
    # forward
    x_feature, x_rec = G(z_t)  # (Nt, 2) + imSize
    x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, 1) + imSize
    x_rec = torch.reshape(x_rec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
    k_rec = fft2c_norm_torch(S_t * x_rec)
    # loss
    lossDC = lossFunc_G(C2R_insert_torch(M_t * k_rec), C2R_insert_torch(M_t * y_G))
    # total loss
    lossAll = lossDC
    # backward
    lossAll.backward()
    # optimize weight
    optimizer_G.step()
    # ------------------------------------------------------------------------------------------------------------------ evaluation
    if ((iter + 1) % eval_every_G == 0):
        G.eval()
        _, imgRec = G(z_t)
        imgRec = R2C_cat_torch(imgRec, axis=1)  # (Nt, 1) + imSize
        imgRec = torch.reshape(imgRec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
        imgRec_np = imgRec.detach().cpu().numpy()
        # normalize
        imgRec_np = np.abs(imgRec_np)
        imgRec_np = imgRec_np / np.max(np.abs(imgRec_np))
        # metrics
        nRMSE = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='nRMSE'))
        PSNR = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='PSNR'))
        SSIM = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='SSIM'))
        # print
        logStr1 = "iter: {:d}  |  time: {:.1f}  |  loss: {:.1f}".format(
            iter + 1, time.time() - tic_iter, lossAll.item())
        logStr2 = "------ nRMSE: {:.4f}  |  PSNR: {:.2f}  |  SSIM: {:.4f}".format(
            nRMSE, PSNR, SSIM)
        logStr = logStr1 + '\n' + logStr2
        print(logStr)
        with open(log_G_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
# ------ save G
saveName = 'G' + ('_it' + str(Niter_G))
savePath = os.path.join(G_Folder, saveName + '.pt')
torch.save(G.state_dict(), savePath)
# ======================================================================================================================
# ====================================================================================================================== pretrain C
# ======================================================================================================================
# ------ load pretrained G
G_pretrain = 'G' + ('_it' + str(Niter_G)) + '.pt'
G.load_state_dict(torch.load(os.path.join(G_Folder, G_pretrain), map_location=device))
# ------ normalize k-space data
G.eval()
C.eval()
with torch.no_grad():
    xft, _ = G(z_t)  # (Nt, 2) + imSize
    xtmp = C(X=xft, nn_idx=None)
    xtmp = R2C_cat_torch(xtmp, axis=1)  # (Nt, 1) + imSize
    xtmp = torch.reshape(xtmp, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
    scaling_factor_C = np.linalg.norm(np.mean(xtmp.detach().cpu().numpy(),axis=1)) / np.linalg.norm(np.mean(AHy_np,axis=1))
y_C = y_t * scaling_factor_C
# save scaling factor
save_dict_h5(h5_path=os.path.join(C_Folder,'scaling_factor_C.h5'), np_dict={'scaling_factor_C': float(scaling_factor_C)})
# ------ pretrain C
# loss
lossFunc_C = nn.MSELoss(reduction='sum')
# optimizer
optimizer_C = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, C.parameters()), 'lr': lr_C}
], betas=betas_C)
# train
tic_iter = time.time()
for iter in range(Niter_C):
    G.eval()
    C.train()
    optimizer_C.zero_grad()
    # forward
    x_feature, _ = G(z_t)  # (Nt, 2) + imSize
    x_rec = C(X=x_feature, nn_idx=None)
    x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, 1) + imSize
    x_rec = torch.reshape(x_rec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
    k_rec = fft2c_norm_torch(S_t * x_rec)
    # loss
    lossDC = lossFunc_C(C2R_insert_torch(M_t * k_rec), C2R_insert_torch(M_t * y_C))
    # total loss
    lossAll = lossDC
    # backward
    lossAll.backward()
    # optimize weight
    optimizer_C.step()
    # ------------------------------------------------------------------------------------------------------------------ evaluation
    if ((iter + 1) % eval_every_C == 0):
        G.eval()
        C.eval()
        imgFT, _ = G(z_t)
        imgRec = C(X=imgFT, nn_idx=None)
        imgRec = R2C_cat_torch(imgRec, axis=1)  # (Nt, 1) + imSize
        imgRec = torch.reshape(imgRec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
        imgRec_np = imgRec.detach().cpu().numpy()
        # normalize
        imgRec_np = np.abs(imgRec_np)
        imgRec_np = imgRec_np / np.max(np.abs(imgRec_np))
        # metrics Rec
        nRMSE = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='nRMSE'))
        PSNR = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='PSNR'))
        SSIM = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='SSIM'))
        # print
        logStr1 = "iter: {:d}  |  time: {:.1f}  |  loss: {:.1f}".format(
            iter + 1, time.time() - tic_iter, lossAll.item())
        logStr2 = "------ nRMSE: {:.4f}  |  PSNR: {:.2f}  |  SSIM: {:.4f}".format(
            nRMSE, PSNR, SSIM)
        logStr = logStr1 + '\n' + logStr2
        print(logStr)
        with open(log_C_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
# ------ save
saveName = 'C' + ('_it' + str(Niter_C))
savePath = os.path.join(C_Folder, saveName + '.pt')
torch.save(C.state_dict(), savePath)
# ======================================================================================================================
# ====================================================================================================================== fine-tune as the final stage of pretrain
# ======================================================================================================================
# ------ load pretrained G and C
G_pretrain = 'G' + ('_it' + str(Niter_G)) + '.pt'
C_pretrain = 'C' + ('_it' + str(Niter_C)) + '.pt'
G.load_state_dict(torch.load(os.path.join(G_Folder, G_pretrain), map_location=device))
C.load_state_dict(torch.load(os.path.join(C_Folder, C_pretrain), map_location=device))
# ------ load scaling factor
scaling_factor_tune = read_dict_h5(os.path.join(C_Folder,'scaling_factor_C.h5'))['scaling_factor_C']
# ------ normalize k-space data
y_tune = y_t * scaling_factor_tune
# save scaling factor
save_dict_h5(h5_path=os.path.join(tune_Folder,'scaling_factor_tune.h5'), np_dict={'scaling_factor_tune': float(scaling_factor_tune)})
# ------ finetune G and C
# loss
lossFunc_tune = nn.MSELoss(reduction='sum')
# optimizer
optimizer_tune = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_tune},
    {'params': filter(lambda p: p.requires_grad, C.parameters()), 'lr': lr_tune},
], betas=betas_tune)
# variable for nn_idx
nn_idx_fix = np.zeros((Nt, K), dtype=np.int64)
# train
tic_iter = time.time()
for iter in range(Niter_tune):
    G.train()
    C.train()
    optimizer_tune.zero_grad()
    # forward
    x_feature, x_pre = G(z_t)  # (Nt, 2) + imSize
    # train graph or use fixed graph
    if ((iter + 1) <= Niter_graph):
        x_rec = C(X=x_feature, nn_idx=None)
    else:
        x_rec = C(X=x_feature, nn_idx=torch.tensor(nn_idx_fix, dtype=torch.int64).to(device))
    # recon image
    x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, 1) + imSize
    x_rec = torch.reshape(x_rec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
    k_rec = fft2c_norm_torch(S_t * x_rec)
    # loss
    lossDC = lossFunc_tune(C2R_insert_torch(M_t * k_rec), C2R_insert_torch(M_t * y_tune))
    # total loss
    lossAll = lossDC
    # backward
    lossAll.backward()
    # optimize weight
    optimizer_tune.step()
    # ------------------------------------------------------------------------------------------------------------------ evaluation
    if ((iter + 1) % eval_every_tune == 0):
        G.eval()
        C.eval()
        imgFT, _ = G(z_t)
        if ((iter + 1) <= Niter_graph):
            imgRec = C(X=imgFT, nn_idx=None)
        else:
            imgRec = C(X=imgFT, nn_idx=torch.tensor(nn_idx_fix, dtype=torch.int64).to(device))
        # recon image
        imgRec = R2C_cat_torch(imgRec, axis=1)  # (Nt, 1) + imSize
        imgRec = torch.reshape(imgRec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
        imgRec_np = imgRec.detach().cpu().numpy()
        # normalize
        imgRec_np = np.abs(imgRec_np)
        imgRec_np = imgRec_np / np.max(np.abs(imgRec_np))
        # metrics Rec
        nRMSE = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='nRMSE'))
        PSNR = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='PSNR'))
        SSIM = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='SSIM'))
        # print
        logStr1 = "iter: {:d}  |  time: {:.1f}  |  loss: {:.1f}".format(
            iter + 1, time.time() - tic_iter, lossAll.item())
        logStr2 = "Rec ------ nRMSE: {:.4f}  |  PSNR: {:.2f}  |  SSIM: {:.4f}".format(
            nRMSE, PSNR, SSIM)
        logStr = logStr1 + '\n' + logStr2
        print(logStr)
        with open(log_tune_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
    # ------ assuming the best graph structure is determined, fix it
    if ((iter + 1) == Niter_graph):
        G.eval()
        C.eval()
        graphFT, _ = G(z_t)
        X_feat = C.FEN(x=graphFT)
        nn_idx = C.KNN(x=X_feat)
        # save and load
        save_dict_h5(
            h5_path=nn_idx_path,
            np_dict={'nn_idx': nn_idx.detach().cpu().numpy()},
        )
        nn_idx_fix = read_dict_h5(
            h5_path=nn_idx_path,
        )['nn_idx']
# ------ save the finally pretrained weight
GName = 'G' + ('_it' + str(Niter_tune))
CName = 'C' + ('_it' + str(Niter_tune))
torch.save(G.state_dict(), os.path.join(tune_Folder, GName + '.pt'))
torch.save(C.state_dict(), os.path.join(tune_Folder, CName + '.pt'))


