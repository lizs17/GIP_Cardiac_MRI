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
parser.add_argument("--R", type=float, default=16.0)
parser.add_argument("--ACS1", type=int, default=6)
parser.add_argument("--ACS2", type=int, default=6)
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
Niter_pretrain = 15000
# ------ ADMM parameters
rho = 0.001
Niter_ADMM = 20
Niter_X = 10
lr_Gz = 1e-5
Niter_Gz = 500
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
postfix = ('_R' + str(R)) + ('_ACS' + str(ACS1) + 'x' + str(ACS2))
reconFolder = 'GIP_Poisson' + postfix + '/'
# ------
caseFolder = os.path.join(reconFolder, dataName + '/')
tune_Folder = os.path.join(caseFolder, 'tune/')
# ------
nn_idx_path = os.path.join(tune_Folder, 'nn_idx.h5')
# ------
ADMM_Folder = os.path.join(caseFolder, 'ADMM/')
if not os.path.exists(ADMM_Folder):
    os.mkdir(ADMM_Folder)
log_ADMM_Path = os.path.join(ADMM_Folder, 'log_ADMM.txt')
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
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
z_Path = os.path.join(caseFolder, 'z.h5')
z_np = read_dict_h5(z_Path)['z']
z_t = torch.tensor(z_np, dtype=dtype_float_torch).to(device)
# ------
G = generator_G(device=device, imSize=imSize, Nt=Nt, ch=CH, ch_z=z_t.size(1), ch_out=2)
G.to(device)
G.weight_init()
# ------
C = generator_GCN(device=device, imSize=imSize, Nt=Nt, K=K, CH=CH, ch_out=2).to(device)
C.to(device)
C.weight_init()
# ======================================================================================================================
# ====================================================================================================================== ADMM
# ======================================================================================================================
# ------ load pretrained weight
G_pretrain = 'G' + ('_it' + str(Niter_pretrain)) + '.pt'
C_pretrain = 'C' + ('_it' + str(Niter_pretrain)) + '.pt'
G.load_state_dict(torch.load(os.path.join(tune_Folder, G_pretrain), map_location=device))
C.load_state_dict(torch.load(os.path.join(tune_Folder, C_pretrain), map_location=device))
# ------ load scaling factor
scaling_factor_ADMM = read_dict_h5(os.path.join(tune_Folder,'scaling_factor_tune.h5'))['scaling_factor_tune']
# ------ normalize k-space data
y_ADMM = y_t * scaling_factor_ADMM
AHy_t = torch.sum(torch.conj(S_t) * ifft2c_norm_torch(y_ADMM), dim=0, keepdim=True)  # (1, Nt, 1) + imSize
# save scaling factor
save_dict_h5(h5_path=os.path.join(ADMM_Folder,'scaling_factor_ADMM.h5'), np_dict={'scaling_factor_ADMM': float(scaling_factor_ADMM)})
# ------ load graph adjacent matrix
nn_idx_fix = read_dict_h5(h5_path=nn_idx_path)['nn_idx']
nn_idx = torch.tensor(nn_idx_fix, dtype=torch.int64).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- initial value
# ----------------------------------------------------------------------------------------------------------------------
G.eval()
C.eval()
xft, xpre = G(z_t)  # (Nt, 2) + imSize
Gz0 = C(X=xft, nn_idx=nn_idx)
Gz0 = R2C_cat_torch(Gz0, axis=1)  # (Nt, 1) + imSize
Gz = torch.reshape(Gz0, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
# ------
X = Gz
# ------
Gam = torch.zeros((1, Nt, 1) + imSize, dtype=dtype_complex_torch).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
lossFunc_ADMM = nn.MSELoss(reduction='sum')
optimizer_ADMM = torch.optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_Gz},
    {'params': filter(lambda p: p.requires_grad, C.parameters()), 'lr': lr_Gz},
], betas=(0.5, 0.98))
# ------
def AHAop(x):
    out = M_t * fft2c_norm_torch(S_t * x)
    out = torch.sum(torch.conj(S_t) * ifft2c_norm_torch(out), dim=0, keepdim=True)
    return out
Normal_op = lambda x: AHAop(x) + rho * x
# ------
tic_iter = time.time()
for iter_ADMM in range(Niter_ADMM):
    # ============ solve subproblem 1 ============
    with torch.no_grad():
        b = AHy_t + rho * Gz - Gam
        r = b - Normal_op(X)
        p = r
        rHr = torch.sum(torch.abs(r) ** 2).item()
        for iter_sub1 in range(Niter_X):
            # gradient descent
            Bp = Normal_op(p)
            alpha = rHr / torch.abs(torch.sum(torch.conj(p) * Bp)).item()
            X = X + alpha * p
            r = r - alpha * Bp
            rHr_new = torch.sum(torch.abs(r) ** 2).item()
            # conjugate gradient
            beta = rHr_new / rHr
            p = r + beta * p
            rHr = rHr_new
        # --- eval
        imgRec_np = X.detach().cpu().numpy()
        # normalize
        imgRec_np = np.abs(imgRec_np)
        imgRec_np = imgRec_np / np.max(np.abs(imgRec_np))
        # metrics Rec
        nRMSE_Rec = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='nRMSE'))
        PSNR_Rec = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='PSNR'))
        SSIM_Rec = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgRec_np, name='SSIM'))
        # print
        logStr = "----------------------------------------------------------\n"
        logStr = logStr + "iter_ADMM: {:d}  |  CG_X  |  time: {:.1f}  |  nRMSE: {:.4f}  |  PSNR: {:.2f}  |  SSIM: {:.4f}".format(
            iter_ADMM + 1, time.time() - tic_iter, nRMSE_Rec, PSNR_Rec, SSIM_Rec)
        print(logStr)
        with open(log_ADMM_Path, 'a') as f:
            f.write(logStr)
            f.write('\n')
    # ============ solve subproblem 2 ============
    G.train()
    C.train()
    for iter_Gz in range(Niter_Gz):
        optimizer_ADMM.zero_grad()
        # forward
        x_feature, x_pre = G(z_t)  # (Nt, 2) + imSize
        x_rec = C(X=x_feature, nn_idx=nn_idx)
        x_rec = R2C_cat_torch(x_rec, axis=1)  # (Nt, 1) + imSize
        x_rec = torch.reshape(x_rec, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
        # loss
        loss = lossFunc_ADMM(C2R_insert_torch(x_rec), C2R_insert_torch(X + Gam / rho))
        # backward
        loss.backward()
        # optimize weight
        optimizer_ADMM.step()
        # evaluation
        if (iter_Gz + 1) % 100 == 0:
            imgGz_np = x_rec.detach().cpu().numpy()
            # normalize
            imgGz_np = np.abs(imgGz_np)
            imgGz_np = imgGz_np / np.max(np.abs(imgGz_np))
            # metrics Gz
            nRMSE_Gz = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgGz_np, name='nRMSE'))
            PSNR_Gz = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgGz_np, name='PSNR'))
            SSIM_Gz = np.mean(metrics2D_np(y=imgRef_np, y_pred=imgGz_np, name='SSIM'))
            # print
            logStr = "iter_ADMM: {:d}  |  Gz_iter: {:d}  |  time: {:.1f}  |  nRMSE: {:.4f}  |  PSNR: {:.2f}  |  SSIM: {:.4f}".format(
                iter_ADMM + 1, iter_Gz + 1, time.time() - tic_iter, nRMSE_Gz, PSNR_Gz, SSIM_Gz)
            print(logStr)
            with open(log_ADMM_Path, 'a') as f:
                f.write(logStr)
                f.write('\n')
    GName = 'G' + ('_rec' + str(iter_ADMM + 1))
    CName = 'C' + ('_rec' + str(iter_ADMM + 1))
    torch.save(G.state_dict(), os.path.join(ADMM_Folder, GName + '.pt'))
    torch.save(C.state_dict(), os.path.join(ADMM_Folder, CName + '.pt'))
    # ============ update Gz ============
    G.eval()
    C.eval()
    with torch.no_grad():
        xtempft, _ = G(z_t)  # (Nt, 2) + imSize
        Gz = C(X=xtempft, nn_idx=nn_idx)
        Gz = R2C_cat_torch(Gz, axis=1)  # (Nt, 1) + imSize
        Gz = torch.reshape(Gz, (1, Nt, 1) + imSize)  # (1, Nt, 1) + imSize
    # ============ update Lagrangian Multiplier ============
    Gam = Gam + rho * (X - Gz)
    # ============ save result ============
    saveName = (dataName) + ('_rec' + str(iter_ADMM + 1))
    imgX = X.detach().cpu().numpy()
    imgX = np.reshape(reverse_order_np(imgX), (Nx, Ny, Nt))
    imgGz = Gz.detach().cpu().numpy()
    imgGz = np.reshape(reverse_order_np(imgGz), (Nx, Ny, Nt))
    save_dict_h5(
        h5_path=os.path.join(ADMM_Folder, saveName + '.h5'),
        np_dict={
            'imgX': imgX,
            'imgGz': imgGz,
        }
    )



