import math
import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,RadioButtons
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
import torchkbnufft
from torchkbnufft import KbNufftAdjoint, KbNufft




def getFolderFileNames(folder, postfix, if_sorted=True):
    nameList = os.listdir(folder)
    outList = []
    for i in range(len(nameList)):
        fileName = nameList[i]
        fileType = fileName.split('.')[-1]
        if (('.'+fileType) == postfix):
            outList.append(fileName)
    # sort
    if if_sorted:
        outList = sorted(outList)
    return outList

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))



#region visualization tool

class imshow3D():

    def __init__(self, img):
        if len(img.shape) != 3:
            raise ValueError('Input for imshow3D should be 3D image !')
        self.img = img
        self.vmin = np.min(img)
        self.vmax = np.max(img)
        self.orient = 'A'
        self.N1, self.N2, self.N3 = int(self.img.shape[0]), int(self.img.shape[1]), int(self.img.shape[2])
        # main plot window
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2, left=0.3)
        self.ax.axis('off')
        self.ax.imshow(img[:, :, 0], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        # button for slice orientation
        self.buttonRegion = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.buttonObj = RadioButtons(
            self.buttonRegion,
            ('A', 'S', 'C'),
            active=0
        )
        self.buttonObj.on_clicked(self.buttonCallback)
        # slider region
        self.sliderRegion = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        # initialize by call buttonCallback
        self.buttonCallback(self.orient)
        # show plot
        plt.show()

    def changeImageOnAx(self, sliceID):
        # extract image slice
        if self.orient == 'A':
            imgSlc = np.squeeze(self.img[:, :, sliceID])
        elif self.orient == 'S':
            imgSlc = np.squeeze(self.img[:, sliceID, :])
        else:
            imgSlc = np.squeeze(self.img[sliceID, :, :])
        # plot on ax
        self.ax.cla()
        self.ax.imshow(imgSlc, cmap='gray', vmin=self.vmin, vmax=self.vmax)
        self.fig.canvas.draw_idle()
        return

    def sliderCallback(self, val):
        sliceID = int(self.sliderObj.val - 1)
        self.changeImageOnAx(sliceID)

    def buttonCallback(self, label):
        # update slice dimension
        self.orient = label
        if self.orient == 'A':
            curN = self.N3
        elif self.orient == 'S':
            curN = self.N2
        else:
            curN = self.N1
        # change the image
        sliceID = int(np.ceil(curN / 2))
        self.changeImageOnAx(sliceID)
        # change the slider
        self.sliderRegion.cla()
        self.sliderObj = Slider(
            self.sliderRegion, r'$slice$',
            1, curN,
            valinit=sliceID,
            valstep=1
        )
        self.sliderObj.on_changed(self.sliderCallback)

#endregion



#region NUFFT

class ENCop_NUFFT():
    def __init__(self, device, imSize, ktraj, dcf):
        '''
        MRI Spatial Encoding Operator based on NUFFT
            ktraj:
                k-space trajectory
                (1, ndim, Npt)
            dcf:
                density-compensation-function
                (1, 1, Npt)
        '''
        self.device = device
        self.NUFFT_forward = KbNufft(im_size=imSize).to(device)
        self.NUFFT_adjoint = KbNufftAdjoint(im_size=imSize).to(device)
        self.ktraj = ktraj
        self.dcf = dcf
    def Aop(self, x, smaps, masks=None):
        y = self.NUFFT_forward(image=x, omega=self.ktraj, smaps=smaps, norm="ortho")
        if masks is not None:
            y = masks * y
        return y
    def AHop(self, y, smaps, masks=None):
        if masks is not None:
            y = masks * y
        x = self.NUFFT_adjoint(data=y * self.dcf, omega=self.ktraj, smaps=smaps, norm="ortho")
        return x
    def AHAop(self, x, smaps, masks=None):
        # we do not use the Toeplitz technique, since it is not equivalent to applying Aop and AHop sequentially,
        # which leads to some differences for data normalization and parameter-tuning
        Ax = self.Aop(x=x, smaps=smaps, masks=masks)
        AHAx = self.AHop(y=Ax, smaps=smaps, masks=masks)
        return AHAx

#endregion



#region MRI recon basics for numpy

def fft1c_norm_np(x, dim, shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.fft(
                    np.fft.ifftshift(x, axes=dim),
                    axis=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.fft(x, axis=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def ifft1c_norm_np(x, dim, shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(x, axes=dim),
                    axis=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.ifft(x, axis=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def fft2c_norm_np(x, dim=(-2,-1), shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.fft2(
                    np.fft.ifftshift(x, axes=dim),
                    axes=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.fft2(x, axes=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def ifft2c_norm_np(x, dim=(-2,-1), shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(x, axes=dim),
                    axes=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.ifft2(x, axes=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def fftnc_norm_np(x, dim, shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.fftn(
                    np.fft.ifftshift(x, axes=dim),
                    axes=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.fftn(x, axes=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def ifftnc_norm_np(x, dim, shift=True, norm='ortho'):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    if shift:
        y = np.fft.fftshift(
                np.fft.ifftn(
                    np.fft.ifftshift(x, axes=dim),
                    axes=dim, norm=norm
                ),
                axes=dim
            )
    else:
        y = np.fft.ifftn(x, axes=dim, norm=norm)
    # because np fft will convert complex64 to complex128 dtype by default
    y = y.astype(x.dtype)
    return y

def rss_coil_np(x, dim):
    y = np.sqrt(
        np.sum(np.power(np.abs(x), 2), axis=dim, keepdims=True)
    )
    return y

def reverse_order_np(x):
    # reverse the dimension order of np array
    out = np.transpose(x, np.arange(x.ndim-1, -1, -1))
    return out

#endregion



#region MRI recon basics for pytorch

def fft1c_norm_torch(x, dim, shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.fft(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.fft(x, dim=dim, norm=norm)
    return y

def ifft1c_norm_torch(x, dim, shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.ifft(x, dim=dim, norm=norm)
    return y

def fft2c_norm_torch(x, dim=(-2,-1), shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.fft2(x, dim=dim, norm=norm)
    return y

def ifft2c_norm_torch(x, dim=(-2,-1), shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.ifft2(x, dim=dim, norm=norm)
    return y

def fftnc_norm_torch(x, dim, shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.fftn(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.fftn(x, dim=dim, norm=norm)
    return y

def ifftnc_norm_torch(x, dim, shift=True, norm='ortho'):
    if shift:
        y = torch.fft.fftshift(
                torch.fft.ifftn(
                    torch.fft.ifftshift(x, dim=dim),
                    dim=dim, norm=norm
                ),
                dim=dim
            )
    else:
        y = torch.fft.ifftn(x, dim=dim, norm=norm)
    return y

def rss_coil_torch(x, dim):
    y = torch.sqrt(
        torch.sum(torch.pow(torch.abs(x), 2), dim=dim, keepdim=True)
    )
    return y

def reverse_order_torch(x):
    # reverse the dimension order of torch tensor
    N = len(x.size())
    revArr = tuple(reversed([i for i in range(N)]))
    out = torch.permute(x, revArr)
    return out

#endregion



#region coil compression method

def calc_GCC_Mtx(calibDATA, ws):
    '''
    calibDATA: [Nx, ..., Nc]
    ws: window size
    '''
    orig_size = calibDATA.shape
    N = len(orig_size)
    Nx = orig_size[0]
    Nc = orig_size[-1]
    SZother = orig_size[1:(N-1)]
    Nother = np.prod(SZother)
    # round sliding window size to nearest odd number
    ws = math.floor(ws/2) * 2 + 1
    # Nsample
    Nsample = ws * Nother
    if Nsample < (3*Nc):
        print('-------------------------- Caution! GCC ws too small ! --------------------------')
    # ifft along fully-sampled dimension
    data = ifft1c_norm_np(calibDATA, dim=0)
    # padding along x direction, which is not inluencing on subspace calculation
    data_padding = zero_padding(data, (Nx+ws-1,)+SZother+(Nc,))
    data_padding = np.reshape(data_padding, (Nx+ws-1,Nother,Nc))
    # calcualte compression matrix for each position along Nx direction
    Nbasis = min(Nc, Nsample)
    GCC_mtx = np.zeros((Nc,Nbasis,Nx), dtype=np.complex64)
    Svals = np.zeros((Nx,Nbasis))
    for xx in range(Nx):
        data_local = np.reshape(data_padding[xx:(xx+ws),:,:], (Nsample,Nc))
        U, S, VH = np.linalg.svd(data_local, full_matrices=False)
        V = VH.conj().T
        GCC_mtx[:,:,xx] = V
        Svals[xx] = S
    return GCC_mtx, Svals

def align_GCC_Mtx(mtx):
    '''
    mtx: [Nc,Vc,Nx]
    '''
    GCC_mtx = mtx.copy()
    Nc,Vc,Nx = GCC_mtx.shape
    # align everything based on the middle slice
    x0 = math.floor(Nx/2) - 1
    A00 = GCC_mtx[:,:,x0]
    # Align backwards to first slice
    A0 = A00.copy()
    for x in range(x0-1, -1, -1):
        A1 = GCC_mtx[:,:,x]
        # calculate transform P
        C = np.matmul(A1.conj().T, A0)
        U, S, VH = np.linalg.svd(C, full_matrices=False)
        V = VH.conj().T
        P = np.matmul(V, U.conj().T)
        # a more smooth compression for this slice
        GCC_mtx[:,:,x] = np.matmul(A1, P.conj().T)
        # iteration
        A0 = GCC_mtx[:,:,x]
    # Align forward to end slice
    A0 = A00.copy()
    for x in range(x0+1, Nx, 1):
        A1 = GCC_mtx[:,:,x]
        C = np.matmul(A1.conj().T, A0)
        U, S, VH = np.linalg.svd(C, full_matrices=False)
        V = VH.conj().T
        P = np.matmul(V, U.conj().T)
        GCC_mtx[:,:,x] = np.matmul(A1, P.conj().T)
        A0 = GCC_mtx[:,:,x]
    return GCC_mtx

def compress_by_GCC_Mtx(DATA, GCC_mtx):
    '''
        DATA:
            [Nx, ..., Nc]
        GCC_mtx:
            [Nc, Vc, Nx]
    '''
    Nc, Vc, Nx = GCC_mtx.shape
    data_size = DATA.shape
    assert (data_size[0]==Nx and data_size[-1]==Nc)
    N = len(data_size)
    SZother = data_size[1:(N-1)]
    Nother = np.prod(SZother)
    # reshape data
    data = np.reshape(DATA, (Nx,Nother,Nc))
    data = ifft1c_norm_np(data, dim=0)
    # Compress data at each x position
    out = np.zeros((Nx,Nother,Vc), dtype=np.complex64)
    for x in range(Nx):
        out[x,:,:] = np.matmul(data[x,:,:], GCC_mtx[:,:,x])
    # reshape data back
    out = fft1c_norm_np(out, dim=0)
    out = np.reshape(out, (Nx,)+SZother+(Vc,))
    return out

def coil_GCC(data_raw, data_calib, Vc, gcc_window=5, if_analysis=False):
    '''
        data_raw:
            [Nx, ..., Nc]
        data_calib:
            [Nx, ..., Nc]
    '''
    # GCC_mtx: [Nc,Nbasis,Nx]   Svals: [Nx,Nbasis]
    GCC_mtx, Svals = calc_GCC_Mtx(data_calib, gcc_window)
    # reduce coil number
    GCC_mtx_cropped = GCC_mtx[:,0:Vc,:]
    # alignment
    GCC_mtx_aligned = align_GCC_Mtx(GCC_mtx_cropped)
    #
    if_out_list = isinstance(data_raw, list)
    if not if_out_list:
        data_raw = [data_raw]
    # compress the data
    data_out = []
    for ii in range(len(data_raw)):
        data_out.append(compress_by_GCC_Mtx(data_raw[ii], GCC_mtx_aligned))
    #
    if not if_out_list:
        data_out = data_out[0]
    return data_out

def coil_SVD(data_raw, data_calib, Vc=None, svalThresh=None, if_analysis=False):
    '''
    Input:
        data_raw:
            [..., Nc]
        data_calib:
            [..., Nc]
    Output:
        data_cc:
            [..., Vc]
    '''
    assert data_raw.shape[-1] == data_calib.shape[-1]
    Nc = data_raw.shape[-1]
    # calculate subspace along coil channel dimension
    calibSZ = data_calib.shape[:-1]
    Ncalib = int(np.prod(calibSZ))
    ctmp = np.reshape(data_calib, (Ncalib,Nc))
    U, S, VH = np.linalg.svd(ctmp, full_matrices=False)
    V = VH.conj().T
    # determine Vc (number of virtual coil channels)
    if Vc is None:
        if svalThresh is None:
            svalThresh = 0.05
        rel_S = S / S[0]
        Vc = np.argmax(rel_S > svalThresh)
        Vc = Vc + 1  # because python index start from 0
    # determine subspace
    Vproj = V[:,0:Vc]
    # coil compression
    rawSZ = data_raw.shape[:-1]
    Nraw = int(np.prod(rawSZ))
    data_cc = np.reshape(data_raw, (Nraw,Nc))
    data_cc = np.matmul(data_cc, Vproj)
    data_cc = np.reshape(data_cc, rawSZ+(Vc,))
    return data_cc

#endregion



#region center crop and padding

def center_range(L, W):
    '''
        center range of W in L
    '''
    start = L//2 - W//2
    end = start + W
    return (start, end)

def get_center_slice_idx_tuple(center_size, all_size):
    assert len(center_size) == len(all_size)
    N = len(all_size)
    for i in range(N):
        assert center_size[i]<=all_size[i]
    # get slice indexes
    idx = []
    for i in range(N):
        dim_range = center_range(all_size[i], center_size[i])
        idx += [slice(dim_range[0], dim_range[1])]
    idx = tuple(idx)
    return idx

def center_crop(x, out_size):
    assert len(x.shape) == len(out_size)
    N = len(out_size)
    for i in range(N):
        assert x.shape[i]>=out_size[i]
    # center idx
    idx = get_center_slice_idx_tuple(center_size=out_size, all_size=x.shape)
    # crop the center
    out = x[idx]
    return out

def number_padding(x, out_size, num):
    assert len(x.shape)==len(out_size)
    N = len(out_size)
    for i in range(N):
        assert x.shape[i]<=out_size[i]
    # padding values
    out = np.ones(out_size, dtype=x.dtype) * np.array(num, dtype=x.dtype)
    # center idx
    idx = get_center_slice_idx_tuple(center_size=x.shape, all_size=out_size)
    # fill the center
    out[idx] = x
    return out

def zero_padding(x, out_size):
    return number_padding(x, out_size, 0)

#endregion



#region ESPIRiT

def ESPIRiT_general(kdata, ACSSize, kerSize, sThresh_Hankel=0.01, sThresh_trunc_I=0.99):
    '''
    :param kdata:
        k-space data for calibration, with center-region fully-sampled for calibration
        size:  (Nx,Ny,Nz,Nc)
        "Nc=1" for single-coil data
    :param ACSSize:
        auto-calibration area size
        [Cx, Cy, Cz]
    :param kerSize:
        kernel size in k-space
        [Kx, Ky, Kz]
    :param sThresh_Hankel:
        singular value threshold for Hankel matrix
    :param sThresh_trunc_I:
        truncation threshold for eigenvalues "=1" for ESPIRiT operator in image space
    :return:
    '''
    if len(kdata.shape)!=4:
        raise NotImplementedError("Input kdata should be (Nx,Ny,Nz,Nc)")
    # size of calibration region
    if isinstance(ACSSize, int):
        ACSSize = [ACSSize, ACSSize, ACSSize]
    else:
        if len(ACSSize)!=3:
            raise NotImplementedError("ACS size should be C(integer) or [Cx,Cy,Cz]")
    # size of k-space kernel
    if isinstance(kerSize, int):
        kerSize = [kerSize, kerSize, kerSize]
    else:
        if len(kerSize)!=3:
            raise NotImplementedError("kernel size should be K(integer) or [Kx,Ky,Kz]")
    # k-space data shape
    Nx,Ny,Nz,Nc = kdata.shape
    # calibration region shape
    Cx = ACSSize[0] if (ACSSize[0]<=Nx) else Nx
    Cy = ACSSize[1] if (ACSSize[1]<=Ny) else Ny
    Cz = ACSSize[2] if (ACSSize[2]<=Nz) else Nz
    # Hankel matrix kernel size
    Kx = kerSize[0] if (kerSize[0]<=Cx) else Cx
    Ky = kerSize[1] if (kerSize[1]<=Cy) else Cy
    Kz = kerSize[2] if (kerSize[2]<=Cz) else Cz
    NK = Kx * Ky * Kz
    # kdata in calibration area 'C'
    Cx_start = Nx//2 - Cx//2
    Cy_start = Ny//2 - Cy//2
    Cz_start = Nz//2 - Cz//2
    Cx_range = (Cx_start, Cx_start+Cx)
    Cy_range = (Cy_start, Cy_start+Cy)
    Cz_range = (Cz_start, Cz_start+Cz)
    C = kdata[Cx_range[0]:Cx_range[1], Cy_range[0]:Cy_range[1], Cz_range[0]:Cz_range[1], :].astype(np.complex64)
    # construct Hankel matrix
    Neq_x = Cx - Kx + 1
    Neq_y = Cy - Ky + 1
    Neq_z = Cz - Kz + 1
    Neq = Neq_x * Neq_y * Neq_z
    H = np.zeros((Neq, NK*Nc), dtype=np.complex64)
    idx = 0
    for xdx in range(Neq_x):
      for ydx in range(Neq_y):
          for zdx in range(Neq_z):
              block = C[xdx:xdx+Kx, ydx:ydx+Ky, zdx:zdx+Kz, :]
              H[idx, :] = block.flatten()
              idx = idx + 1
    # Take the Singular Value Decomposition
    U, S, VH = np.linalg.svd(H, full_matrices=True)
    V = VH.conj().T
    # Choose the local subspace in k-space
    rankH = np.sum(S >= sThresh_Hankel * S[0])
    V = V[:, 0:rankH]
    # ==============================  Part 1: build i-space ESPIRiT  ==============================
    # get subspace in k-space: fill V into a convolution kernel
    Kx_start = Nx // 2 - Kx // 2
    Ky_start = Ny // 2 - Ky // 2
    Kz_start = Nz // 2 - Kz // 2
    Kx_range = (Kx_start, Kx_start+Kx)
    Ky_range = (Ky_start, Ky_start+Ky)
    Kz_range = (Kz_start, Kz_start + Kz)
    subspace_kspc = np.zeros((Nx,Ny,Nz,Nc,rankH), dtype=np.complex64)
    for idx in range(rankH):
        subspace_kspc[Kx_range[0]:Kx_range[1], Ky_range[0]:Ky_range[1], Kz_range[0]:Kz_range[1], :, idx] = np.reshape(V[:, idx], (Kx,Ky,Kz,Nc))
    # get subspace in i-space: transform to image space
    subspace_ispc = np.zeros((Nx,Ny,Nz,Nc,rankH), dtype=np.complex64)
    for idx in range(rankH):
        for jdx in range(Nc):
            ker = subspace_kspc[::-1, ::-1, ::-1, jdx, idx].conj()
            subspace_ispc[:,:,:,jdx,idx] = fftnc_norm_np(ker, (0,1,2)) * np.sqrt(Nx*Ny*Nz)/np.sqrt(Kx*Ky*Kz)
    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than 'sThresh_Esp_I'
    subspace_ispc_truncate = np.zeros((Nx,Ny,Nz,Nc,Nc), dtype=np.complex64)
    for xdx in range(Nx):
        for ydx in range(Ny):
            for zdx in range(Nz):
                Gq = subspace_ispc[xdx,ydx,zdx,:,:]
                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for cdx in range(Nc):
                    if (s[cdx]**2 > sThresh_trunc_I):
                        subspace_ispc_truncate[xdx,ydx,zdx,:,cdx] = u[:,cdx]
    # sensitivity map
    sensMap = subspace_ispc_truncate
    # ESPIRiT operator in i-space (permute to the format ESP * x)
    ESP_ispc = np.matmul(
        subspace_ispc.conj(),
        np.transpose(subspace_ispc,(0,1,2,4,3)))
    ESP_ispc_truncate = np.matmul(
        subspace_ispc_truncate.conj(),
        np.transpose(subspace_ispc_truncate,(0,1,2,4,3)))
    # ==============================  Part 2: build k-space ESPIRiT  ==============================
    # extract the projection weight for the center element in the convolution kernel
    tmp = V.conj()
    projection_op = np.matmul(tmp.conj(), tmp.T)
    center_start = (Kx // 2) * (Ky * Kz * Nc) + (Ky // 2) * (Kz * Nc) + (Kz // 2) * (Nc)
    projection_op_center = projection_op[:, center_start:(center_start+Nc)]
    # k-space local kernel weight
    ESP_kspc = np.zeros((Nc, Nc, Kx, Ky, Kz), dtype=np.complex64)
    for cc in range(Nc):
        ker_weight = np.reshape(projection_op_center[:, cc], (Kx,Ky,Kz,Nc))
        ESP_kspc[cc] = np.transpose(ker_weight, (3, 0, 1, 2))
    # ==============================  Part 3: return result  ==============================
    return sensMap, ESP_ispc, ESP_ispc_truncate, ESP_kspc

def ESPIRiT_2D(kdata, ACSSize, kerSize, sThresh_Hankel=0.01, sThresh_trunc_I=0.99):
    '''
    :param kdata:
        k-space data for calibration, with center-region fully-sampled for calibration
        size:  (Nx,Ny,Nc)
        "Nc=1" for single-coil data
    :param ACSSize:
        auto-calibration area size
        [Cx, Cy]
    :param kerSize:
        kernel size in k-space
        [Kx, Ky]
    :param sThresh_Hankel:
        singular value threshold for Hankel matrix
    :param sThresh_Esp_I:
        crop threshold for eigenvalues "=1" for ESPIRiT operator in image space
    :return:
    '''
    if len(kdata.shape)!=3:
        raise NotImplementedError("Input kdata should be (Nx,Ny,Nc)")
    # size of calibration region
    if isinstance(ACSSize, int):
        ACSSize = [ACSSize, ACSSize]
    else:
        if len(ACSSize)!=2:
            raise NotImplementedError("ACS size should be C(integer) or [Cx,Cy]")
    # size of k-space kernel
    if isinstance(kerSize, int):
        kerSize = [kerSize, kerSize]
    else:
        if len(kerSize)!=2:
            raise NotImplementedError("kernel size should be K(integer) or [Kx,Ky]")
    # shape params
    Nx,Ny,Nc = kdata.shape
    # calibration region shape
    Cx = ACSSize[0] if (ACSSize[0]<=Nx) else Nx
    Cy = ACSSize[1] if (ACSSize[1]<=Ny) else Ny
    # Hankel matrix kernel size
    Kx = kerSize[0] if (kerSize[0]<=Cx) else Cx
    Ky = kerSize[1] if (kerSize[1]<=Cy) else Cy
    NK = Kx * Ky
    # kdata in calibration area 'C'
    Cx_start = Nx//2 - Cx//2
    Cy_start = Ny//2 - Cy//2
    Cx_range = (Cx_start, Cx_start+Cx)
    Cy_range = (Cy_start, Cy_start+Cy)
    C = kdata[Cx_range[0]:Cx_range[1], Cy_range[0]:Cy_range[1], :].astype(np.complex64)
    # construct Hankel matrix
    Neq_x = Cx - Kx + 1
    Neq_y = Cy - Ky + 1
    Neq = Neq_x * Neq_y
    H = np.zeros((Neq, NK*Nc), dtype=np.complex64)
    idx = 0
    for xdx in range(Neq_x):
      for ydx in range(Neq_y):
          block = C[xdx:xdx+Kx, ydx:ydx+Ky, :]
          H[idx, :] = block.flatten()
          idx = idx + 1
    # Take the Singular Value Decomposition
    U, S, VH = np.linalg.svd(H, full_matrices=True)
    V = VH.conj().T
    # Choose the local subspace in k-space
    rankH = np.sum(S >= sThresh_Hankel * S[0])
    V = V[:, 0:rankH]
    # ==============================  Part 1: build i-space ESPIRiT  ==============================
    # get subspace in k-space: fill V into a convolution kernel
    Kx_start = Nx//2 - Kx//2
    Ky_start = Ny//2 - Ky//2
    Kx_range = (Kx_start, Kx_start+Kx)
    Ky_range = (Ky_start, Ky_start+Ky)
    subspace_kspc = np.zeros((Nx,Ny,Nc,rankH), dtype=np.complex64)
    for idx in range(rankH):
        subspace_kspc[Kx_range[0]:Kx_range[1], Ky_range[0]:Ky_range[1], :, idx] = np.reshape(V[:, idx], (Kx,Ky,Nc))
    # get subspace in i-space: transform to image space
    subspace_ispc = np.zeros((Nx,Ny,Nc,rankH), dtype=np.complex64)
    for idx in range(rankH):
        for jdx in range(Nc):
            ker = subspace_kspc[::-1, ::-1, jdx, idx].conj()
            subspace_ispc[:,:,jdx,idx] = fftnc_norm_np(ker, (0,1)) * np.sqrt(Nx*Ny)/np.sqrt(Kx*Ky)
    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than 'sThresh_Esp_I'
    subspace_ispc_truncate = np.zeros((Nx,Ny,Nc,Nc), dtype=np.complex64)
    for xdx in range(Nx):
        for ydx in range(Ny):
            Gq = subspace_ispc[xdx,ydx,:,:]
            u, s, vh = np.linalg.svd(Gq, full_matrices=True)
            for cdx in range(Nc):
                if (s[cdx]**2 > sThresh_trunc_I):
                    subspace_ispc_truncate[xdx,ydx,:,cdx] = u[:,cdx]
    # sensitivity map
    sensMap = subspace_ispc_truncate
    # ESPIRiT operator in i-space
    ESP_ispc = np.matmul(
        subspace_ispc.conj(),
        np.transpose(subspace_ispc,(0,1,3,2)))
    ESP_ispc_truncate = np.matmul(
        subspace_ispc_truncate.conj(),
        np.transpose(subspace_ispc_truncate,(0,1,3,2)))
    # ==============================  Part 2: build k-space ESPIRiT  ==============================
    # extract the projection weight for the center element in the convolution kernel
    tmp = V.conj()
    projection_op = np.matmul(tmp.conj(), tmp.T)
    center_start = (Kx//2) * (Ky*Nc) + (Ky//2) * (Nc)
    projection_op_center = projection_op[:, center_start:(center_start+Nc)]
    # k-space local kernel weight
    ESP_kspc = np.zeros((Nc, Nc, Kx, Ky), dtype=np.complex64)
    for cc in range(Nc):
        ker_weight = np.reshape(projection_op_center[:, cc], (Kx,Ky,Nc))
        ESP_kspc[cc] = np.transpose(ker_weight, (2, 0, 1))
    # ==============================  Part 3: return result  ==============================
    return sensMap, ESP_ispc, ESP_ispc_truncate, ESP_kspc

#endregion



#region Real <---> Complex

def R2C_insert_np(x, axis):
    """
    x:    floa32     [...,2,...]
    out:  complex64  [...]
    """
    assert (x.dtype=='float32')
    # first transpose the given axis to the last dim
    N = x.ndim
    newshape = tuple([i for i in range(0, axis)]) + tuple([i for i in range(axis+1, N)]) + (axis,)
    out = np.transpose(x, newshape)
    # then get complex value
    out = out[...,0] + 1j * out[...,1]
    return out

def C2R_insert_np(x, axis):
    """
    x:    complex64  [...]
    out:  floa32     [...,2,...]
    """
    # assert (x.dtype=='complex64')
    # first insert Real/Imag dimension to the last dim
    if x.dtype == np.complex128:
        dtype_np = np.float64
    elif x.dtype == np.complex64:
        dtype_np = np.float32
    else:
        raise NotImplementedError
    out = np.zeros(x.shape+(2,), dtype=dtype_np)
    out[...,0] = x.real
    out[...,1] = x.imag
    # then transpose to given axis
    N = out.ndim
    newshape = tuple([i for i in range(0, axis)]) + (N-1,) + tuple([i for i in range(axis, N-1)])
    out = np.transpose(out, newshape)
    return out

def R2C_insert_torch(x, axis=-1):
    """
    x:    floa32     [...,2,...]
    out:  complex64  [...]
    """
    assert (x.dtype==torch.float32)
    out = x
    # first transpose the given axis to the last dim
    if axis >= 0:
        N = out.ndim
        newshape = tuple([i for i in range(0, axis)]) + tuple([i for i in range(axis+1, N)]) + (axis,)
        out = out.permute(newshape)
        out = out.contiguous()
    # then get complex value
    out = torch.view_as_complex(out)
    return out

def C2R_insert_torch(x, axis=-1):
    """
    x:    complex64  [...]
    out:  floa32     [...,2,...]
    """
    # assert (x.dtype==torch.complex64)
    # first insert Real/Imag dimension to the last dim
    out = torch.view_as_real(x)
    # then transpose to given axis
    if axis >= 0:
        N = out.ndim
        newshape = tuple([i for i in range(0, axis)]) + (N-1,) + tuple([i for i in range(axis, N-1)])
        out = out.permute(newshape)
        out = out.contiguous()
    return out

def R2C_cat_np(x, axis):
    """
    x:    float32
    out:  complex64
    """
    N = x.shape[axis]
    assert (N%2 == 0)
    M = int(N/2)
    if axis == 0:
        out = x[:M] + 1j*x[M:]
    else:
        out = np.swapaxes(x, axis1=axis, axis2=0)
        out = out[:M] + 1j * out[M:]
        out = np.swapaxes(out, axis1=axis, axis2=0)
    return out

def C2R_cat_np(x, axis):
    """
    x:    complex64
    out:  float32
    """
    out = np.concatenate((x.real, x.imag), axis=axis)
    return out

def R2C_cat_torch(x, axis):
    """
    x:    float32
    out:  complex64
    """
    N = x.shape[axis]
    assert (N%2 == 0)
    M = int(N/2)
    if axis == 0:
        out = x[:M] + 1j*x[M:]
    elif axis == 1:
        out = x[:, :M] + 1j*x[:, M:]
    elif axis == 2:
        out = x[:, :, :M] + 1j * x[:, :, M:]
    else:
        raise NotImplementedError
    return out

def C2R_cat_torch(x, axis):
    """
    x:    complex64
    out:  float32
    """
    out = torch.cat((x.real, x.imag), axis=axis)
    return out

#endregion



#region data reading & writing

def save_dict_h5(h5_path, np_dict):
    with h5py.File(h5_path,'w') as f:
        for key, value in np_dict.items():
            if isinstance(value, dict):
                dict_group = f.create_group('key')
                for dict_key, dict_val in value.items():
                    dict_group[dict_key] = dict_val
            else:
                f.create_dataset(key, data=value)

def read_dict_h5(h5_path, keys=None, if_reverse=False):
    '''
    if_reverse:
        for matlab saved .h5 file, the data dimension order is reversed
    '''
    if keys is not None:
        if not isinstance(keys, list):
            raise ValueError('keys should be list type object or None')
    # read .h5 file
    with h5py.File(h5_path,'r') as f:
        dict = {}
        # if not given keys, read all fields
        if keys==None:
            keys = f.keys()
        # read data
        for key in keys:
            if if_reverse:
                dict[key] = reverse_order_np(f[key][()])
            else:
                dict[key] = f[key][()]
    return dict

def mat_v73_reader(mat_v73_path):
    '''
    :param mat_path: .mat file path which is saved by -v7.3 mode
    :return: dict
    '''
    f = h5py.File(mat_v73_path, 'r')
    dict = {}
    for key in f.keys():
        item = np.array(f[key])
        # for complex value data
        try:
            data = item['real'] + 1j * item['imag']
        # real value data
        except:
            data = item
        # reverse dimension order
        dict[key] = reverse_order_np(data)
    return dict

#endregion



#region evaluation metrics for numpy

def metrics2D_np(y, y_pred, name):
    '''
        calculate 2D evaluation metrics in numpy
        return value for each item and the mean value
        Input:
            y: reference image
                [..., Ny, Nx]
            y_pred: predicted image
                [..., Ny, Nx]
        Output:
            out: calculated metrics
                [..., ]
    '''
    assert (y.shape==y_pred.shape)
    SZ_in = y.shape
    Ndim = len(y.shape)
    # select metrics function
    if name=='MSE':
        metric_func = lambda x_gt, x: np.mean(np.abs(x - x_gt) ** 2)
    elif name=='nRMSE':
        metric_func = lambda x_gt, x: normalized_root_mse(x_gt, x, normalization='min-max')
    elif name=='PSNR':
        metric_func = peak_signal_noise_ratio
    elif name=='SSIM':
        metric_func = lambda x_gt, x: structural_similarity(x_gt, x, data_range=1.0)
    elif name=='MAE':
        metric_func = lambda x_gt, x: np.mean(np.abs(x - x_gt))
    else:
        raise NotImplementedError
    # reshape image
    SZ_other = SZ_in[:-2]
    Ny, Nx = SZ_in[-2], SZ_in[-1]
    N_other = int(np.prod(np.array(SZ_other)))
    y = np.reshape(y, (N_other,Ny,Nx))
    y_pred = np.reshape(y_pred, (N_other,Ny,Nx))
    # calculate metrics
    out = np.zeros((N_other,1))
    for ii in range(N_other):
        out[ii] = metric_func(y[ii], y_pred[ii])
    # reshape back
    out = np.reshape(out, SZ_other)
    return out

#endregion



#region Pytorch Complex

def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(
            in_features=in_features, out_features=out_features)
        self.fc_i = nn.Linear(
            in_features=in_features, out_features=out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_i = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


class ComplexConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ComplexConv3d, self).__init__()
        self.conv_r = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_i = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


class ComplexConvTranspose2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_tran_r = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        self.conv_tran_i = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)


class ComplexConvTranspose3d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose3d, self).__init__()
        self.conv_tran_r = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        self.conv_tran_i = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

#endregion


