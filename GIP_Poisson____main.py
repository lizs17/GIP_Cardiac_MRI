import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.transform import resize
# ----------------------------------------------------------------------------------------------------------------------
from utils import reverse_order_np, fft2c_norm_np, ifft2c_norm_np, rss_coil_np
from utils import save_dict_h5, read_dict_h5, imshow3D
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- recon
# ----------------------------------------------------------------------------------------------------------------------
gpu = 0
# ------
dtype_float_np = np.float32
dtype_complex_np = np.complex64
dtype_float_torch = torch.float32
dtype_complex_torch = torch.complex64
# ------
dataName = 'fs_0032_3T_slc1_p3'
R, ACS1, ACS2 = 16.0, 6, 6
outputFolder = 'output_Poisson/'
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)
# ------ pretrain
os.system(
    "python GIP_Poisson____pretrain.py" \
        + " --gpu " + str(gpu) + " --dataName " + str(dataName) \
        + " --R " + str(R) + " --ACS1 " + str(ACS1) + " --ACS2 " + str(ACS2)
)
# ------ ADMM algorithm
os.system(
    "python GIP_Poisson____ADMM.py" \
        + " --gpu " + str(gpu) + " --dataName " + str(dataName) \
        + " --R " + str(R) + " --ACS1 " + str(ACS1) + " --ACS2 " + str(ACS2)
)
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
# ------
maskName = 'maskPoisson' \
           + ('_Nx' + str(Nx) + '_Ny' + str(Ny) + '_Nt' + str(Nt)) \
           + ('_R' + str(R)) + ('_ACS' + str(ACS1) + 'x' + str(ACS2))
masks = read_dict_h5(os.path.join(maskFolder, maskName + '.h5'))['mask']  # (Nx, Ny, Nt)
# ------ dtype
imgGT = imgGT.astype(dtype_complex_np)
smaps = smaps.astype(dtype_complex_np)
masks = masks.astype(dtype_float_np)
# ------ shape: (Nc, Nt, ch, Ny, Nx)
imgGT = np.reshape(reverse_order_np(imgGT), (1, Nt, 1, Ny, Nx))
smaps = np.reshape(reverse_order_np(smaps), (Nc, 1, 1, Ny, Nx))
masks = np.reshape(reverse_order_np(masks), (1, Nt, 1, Ny, Nx))
# ------ normalization
imgGT = imgGT / np.max(np.abs(imgGT))
smaps = smaps / rss_coil_np(smaps, dim=0)
# ------ undersampling
kdata = masks * fft2c_norm_np(smaps * imgGT)  # (Nc, Nt, 1, Ny, Nx)
# ------ zero-filling recon
AHy_np = np.sum(np.conj(smaps) * ifft2c_norm_np(kdata), axis=0, keepdims=True)  # (1, Nt, 1, Ny, Nx)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
imgGT = np.reshape(reverse_order_np(imgGT), (Nx, Ny, Nt))
smaps = np.reshape(reverse_order_np(smaps), (Nx, Ny, Nc))
masks = np.reshape(reverse_order_np(masks), (Nx, Ny, Nt))
imgZF = np.reshape(reverse_order_np(AHy_np), (Nx, Ny, Nt))
# ------
plt.figure(figsize=(8, 4))
for ii in range(8):
    plt.subplot(2, 4, ii + 1)
    plt.imshow(np.abs(smaps[:,:,ii]), cmap='gray')
    plt.axis('off')
    plt.title('coil ' + str(ii + 1))
plt.savefig(os.path.join(outputFolder, 'smaps.png'), bbox_inches='tight')
# ------
plt.figure(figsize=(12, 6))
for ii in range(18):
    plt.subplot(3, 6, ii + 1)
    plt.imshow(masks[:,:,ii], cmap='gray')
    plt.axis('off')
    plt.title('frame ' + str(ii + 1))
plt.savefig(os.path.join(outputFolder, 'masks.png'), bbox_inches='tight')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
postfix = ('_R' + str(R)) + ('_ACS' + str(ACS1) + 'x' + str(ACS2))
reconFolder = 'GIP_Poisson' + postfix + '/'
# ------
caseFolder = os.path.join(reconFolder, dataName + '/')
ADMM_Folder = os.path.join(caseFolder, 'ADMM/')
# ------
imgX_rec = []
imgGz_rec = []
for iter in range(20):
    recName = (dataName) + ('_rec' + str(iter + 1))
    recData = read_dict_h5(os.path.join(ADMM_Folder, recName + '.h5'))
    imgX = recData['imgX']
    imgGz = recData['imgGz']
    imgX = imgX / np.max(imgX)
    imgGz = imgGz / np.max(imgGz)
    imgX_rec.append(imgX)
    imgGz_rec.append(imgGz)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rescale_gray_head(x, head):
    # adjust gray-scale window for better visualization
    out = x / head
    out[out > 1.0] = 1.0
    return out
def plot_recon_figure(imgRC, imgGT):
    # shape: (Nx, Ny, Nt)
    # abs
    imgGT = np.abs(imgGT)
    imgRC = np.abs(imgRC)
    # dtype
    imgGT = imgGT.astype(dtype_float_np)
    imgRC = imgRC.astype(dtype_float_np)
    # ------ error
    imgErr = np.abs(imgRC - imgGT)
    imgGT = rescale_gray_head(imgGT, head=0.6)
    imgRC = rescale_gray_head(imgRC, head=0.6)
    # ------ frame D
    imgFrmD = imgRC[:, :, 16]
    imgFrmD_Err = imgErr[:, :, 16]
    # ------ frame S
    imgFrmS = imgRC[:, :, 8]
    imgFrmS_Err = imgErr[:, :, 8]
    # ------ ROI D
    roiFrmD = imgFrmD[48:114, 59:125]
    roiFrmD_Err = imgFrmD_Err[48:114, 59:125]
    # resize
    roiFrmD = resize(roiFrmD, (Nx, Ny))
    roiFrmD_Err = resize(roiFrmD_Err, (Nx, Ny))
    # ------ ROI S
    roiFrmS = imgFrmS[48:114, 59:125]
    roiFrmS_Err = imgFrmS_Err[48:114, 59:125]
    # resize
    roiFrmS = resize(roiFrmS, (Nx, Ny))
    roiFrmS_Err = resize(roiFrmS_Err, (Nx, Ny))
    # ------ Mmode
    Mmode_gray = np.transpose(np.reshape(imgRC[74, :, :], (Ny, Nt)), (1, 0))  # (Nt, Ny)
    Mmode_err = np.transpose(np.reshape(imgErr[74, :, :], (Ny, Nt)), (1, 0))  # (Nt, Ny)
    # ------ plot figure
    H = 20
    gap = 0.1
    # ------
    unit = (H - gap * (7 + 1)) / (5 * Nx + 2 * Nt)
    W = unit * Ny + gap * 2
    # ------
    w0, h0 = gap / W, gap / H
    unit_percent = unit / H
    # ------
    W_percent = Ny / (Ny + gap * 2)
    H_Mmode_percent = unit_percent * Nt
    H_frm_percent = unit_percent * Nx
    # ------
    h1 = h0
    h2 = h0 + H_Mmode_percent + h0
    h3 = h0 + H_Mmode_percent * 2 + h0 * 2
    h4 = h0 + H_Mmode_percent * 2 + h0 * 2 + H_frm_percent + h0
    h5 = h0 + H_Mmode_percent * 2 + h0 * 2 + H_frm_percent * 2 + h0 * 2
    h6 = h0 + H_Mmode_percent * 2 + h0 * 2 + H_frm_percent * 3 + h0 * 3
    h7 = h0 + H_Mmode_percent * 2 + h0 * 2 + H_frm_percent * 4 + h0 * 4
    # ------
    plt.figure(figsize=(W, H))
    plt.axes([w0, h1, W_percent, H_Mmode_percent]), plt.imshow(Mmode_err, vmin=0.0, vmax=0.04), plt.axis('off')
    plt.axes([w0, h2, W_percent, H_Mmode_percent]), plt.imshow(Mmode_gray, cmap='gray', vmin=0.0, vmax=1.0), plt.axis('off')
    plt.axes([w0, h3, W_percent, H_frm_percent]), plt.imshow(roiFrmS_Err, vmin=0.0, vmax=0.04), plt.axis('off')
    plt.axes([w0, h4, W_percent, H_frm_percent]), plt.imshow(roiFrmS, cmap='gray', vmin=0.0, vmax=1.0), plt.axis('off')
    plt.axes([w0, h5, W_percent, H_frm_percent]), plt.imshow(roiFrmD_Err, vmin=0.0, vmax=0.04), plt.axis('off')
    plt.axes([w0, h6, W_percent, H_frm_percent]), plt.imshow(roiFrmD, cmap='gray', vmin=0.0, vmax=1.0), plt.axis('off')
    plt.axes([w0, h7, W_percent, H_frm_percent]), plt.imshow(imgFrmD, cmap='gray', vmin=0.0, vmax=1.0), plt.axis('off')
# ------
plot_recon_figure(imgRC=imgGT, imgGT=imgGT)
plt.savefig(os.path.join(outputFolder, 'imgGT.png'), bbox_inches='tight')
plot_recon_figure(imgRC=imgZF, imgGT=imgGT)
plt.savefig(os.path.join(outputFolder, 'imgZF.png'), bbox_inches='tight')
plot_recon_figure(imgRC=imgX_rec[0], imgGT=imgGT)
plt.savefig(os.path.join(outputFolder, 'imgRC1.png'), bbox_inches='tight')
plot_recon_figure(imgRC=imgX_rec[9], imgGT=imgGT)
plt.savefig(os.path.join(outputFolder, 'imgRC10.png'), bbox_inches='tight')
plot_recon_figure(imgRC=imgX_rec[19], imgGT=imgGT)
plt.savefig(os.path.join(outputFolder, 'imgRC20.png'), bbox_inches='tight')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
plt.show()





