import torch

# from @caillonantoine https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def safe_log(x):
    return torch.log(x+1e-7)

def multiscale_spectral_loss(a,b,scales,overlap, lin_weight=1.0, log_weight=1.0):
    stfts_a = multiscale_fft(a,scales,overlap)
    stfts_b = multiscale_fft(b,scales,overlap)
    loss = 0
    for s_x, s_y in zip(stfts_a, stfts_b):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss*lin_weight + log_loss* log_weight
    return loss
    
def wasserstein_1d(a,b,p=1):
    cdf_a = torch.cumsum(a,dim=-1)
    cdf_b = torch.cumsum(b,dim=-1)
    if p == 1:
        cdf = torch.sum(torch.abs(cdf_a-cdf_b),dim=-1, keepdim=True)
    if p == 2:
        cdf = torch.sum((cdf_a-cdf_b)**2,dim=-1, keepdim=True)
    assert torch.sum(torch.isnan(cdf)) == 0
    return cdf