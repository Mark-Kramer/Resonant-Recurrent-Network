from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

def rrn_prepare_batch(batch, device):
    # Variable-length speech batches may include lengths as a third item.
    if len(batch) == 3:
        x, y, lengths = batch
        return (
            x.to(device, dtype=torch.float32),
            lengths.to(device, dtype=torch.long),
        ), y.to(device, dtype=torch.long)

    x, y = batch
    return x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)

class RRNModel(nn.Module):

    def __init__(
        self,
        Fs:                    float = 1000,
        n_classes:             int   = 10,
        init_scale:            float = 0.3,
        sigma:                 float = 0.0,
        no_recurrence:         bool  = False,
        rhythm_configuration:  str   = "golden",
        feature_norm:          str   = "batchnorm",
        verbose:               bool  = True
    ):
        super().__init__()
        self.Fs = Fs
        self.no_recurrence = bool(no_recurrence)
        self.rhythm_configuration = str(rhythm_configuration).lower()
        self.verbose = verbose

        # ---------------- Define frequencies from rhythm_configuration ----------------
        f0 = 2.0
        cfg = self.rhythm_configuration
        if cfg == "golden":
            phi     = 1.618
            n_nodes = 40 #11
            frange  = f0 * (phi ** np.arange(0, n_nodes, 1))
        elif cfg == "log":
            phi     = 2.718
            n_nodes = 6
            frange  = f0 * (phi ** np.arange(0, n_nodes, 1))
        elif cfg == "two":
            phi     = 2.0
            n_nodes = 8
            frange  = f0 * (phi ** np.arange(0, n_nodes, 1))
        elif cfg == "linear":
            n_nodes = 11
            frange  = np.linspace(2, 245.9, num=n_nodes)
        else:
            raise ValueError(
                f"Unknown rhythm_configuration='{rhythm_configuration}'. "
                "Choose one of {'golden','log','two','linear'}."
            )

        # ---------------- Compute initial AR(2) weights (w1, w2) ----------------
        r = 0.99999
        
        w_t_minus_1 = 2 * r * np.cos(2 * np.pi * frange / Fs)
        w_t_minus_2 = (-r**2) * np.ones_like(w_t_minus_1)

        # Apply stability constraints
        discriminant      = w_t_minus_1**2 + 4*w_t_minus_2 + 0j
        sqrt_discriminant = np.sqrt(discriminant)
        z1   = (w_t_minus_1 + sqrt_discriminant)/2
        z2   = (w_t_minus_1 - sqrt_discriminant)/2

        r1 = np.abs(z1)
        r2 = np.abs(z2)

        # Ensure |r1| < 1 and |r2| < 1, w_t_minus_1 > 0, and less than Nyquist
        final_valid = (r1 < 1) & (r2 < 1) & (w_t_minus_1 > 0) & (frange < Fs/2)

        # Apply final constraints
        w_t_minus_1 = w_t_minus_1[final_valid]
        w_t_minus_2 = w_t_minus_2[final_valid]
        frange = frange[final_valid]

        if frange.size == 0:
            raise ValueError(
                f"No valid nodes after stability constraints for cfg='{cfg}' (Fs={Fs}, f0={f0}, r={r})."
            )

        self.frange = frange

        if self.verbose:
            print(
                "Rhythm config is ", cfg,
                ", from ", frange[0],
                " to ", np.round(frange[-1]),
                " N nodes=", np.size(frange),
            )
            print("Frequencies: ", np.round(frange))
            self.frange = frange  # stays as numpy; fine

        # Convert AR(2) parameters into buffers
        w1 = torch.tensor(w_t_minus_1.astype(np.float32), dtype=torch.float32)
        w2 = torch.tensor(w_t_minus_2.astype(np.float32), dtype=torch.float32)
        dt = 1.0 / Fs
        a1 = w_t_minus_1
        a2 = w_t_minus_2
        omega = np.sqrt((a1 + a2 - 1) / (a2 * dt**2))
        beta  = -(a1 + 2 * a2) / (2 * a2 * dt)
        omega_t = torch.tensor(omega.astype(np.float32), dtype=torch.float32)
        beta_t  = torch.tensor(beta.astype(np.float32),  dtype=torch.float32)

        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)
        self.register_buffer("omega", omega_t)
        self.register_buffer("beta",  beta_t)

        self.K = self.w1.numel()

        # Normalization over node features
        self.feature_norm = feature_norm
        self.norm = nn.BatchNorm1d(self.K, affine=True, track_running_stats=True)

        # Linear readout: mean amplitude power (K) -> class logits
        self.readout = nn.Linear(self.K, n_classes, bias=True)

        # Input weights: broadcast scalar input to each node
        self.register_buffer("W_in", torch.ones(self.K, dtype=torch.float32))

        # Trainable recurrent matrix W_res (K x K), zero diagonal
        W0 = (torch.rand(self.K, self.K, dtype=torch.float32) - 0.5) * 2.0 * init_scale
        W0.fill_diagonal_(0.0)
        self.W_res = nn.Parameter(W0)

        # Option to remove recurrence 
        if self.no_recurrence:
            with torch.no_grad():
                self.W_res.data.zero_()
            self.W_res.requires_grad_(False)

        # Diagonal mask (no self-connections)
        mask = torch.ones(self.K, self.K, dtype=torch.float32)
        mask.fill_diagonal_(0.0)
        self.register_buffer("mask_nodiag", mask)

        # Training noise (use 0 for stable gradients; turn on later if desired)
        self.sigma = float(sigma)

    @torch.no_grad()
    def project_W_no_diag(self):
        # If recurrence is disabled, nothing to project
        if self.no_recurrence or (not self.W_res.requires_grad):
            return
        self.W_res.data.mul_(self.mask_nodiag)

    def forward(self, seq_batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        seq_batch: (B, T) scalar input per time step, or (seq_batch, lengths)
                   where lengths is (B,) for variable-length padded batches.
        Returns logits of shape (B, n_classes).
        """
        lengths = None
        if isinstance(seq_batch, (tuple, list)):
            seq_batch, lengths = seq_batch

        B, T = seq_batch.shape
        Fs = float(self.Fs)
        dev = seq_batch.device
        if lengths is not None:
            lengths = lengths.to(dev, dtype=torch.long)

        # zero-state on the same device as input
        x_t         = torch.zeros(B, self.K, device=dev)
        x_t_minus_1 = torch.zeros(B, self.K, device=dev)
        x_t_minus_2 = torch.zeros(B, self.K, device=dev)
        amp_power_accum = torch.zeros(B, self.K, device=dev)

        for t in range(T):
            u_t         = seq_batch[:, t]  # (B,)
            input_drive = u_t.unsqueeze(1) * self.W_in.unsqueeze(0)  # (B, K)
            hist_drive  = self.w1 * x_t_minus_1 + self.w2 * x_t_minus_2

            if self.no_recurrence:
                recur_drive = 0.0
            else:
                gated = torch.tanh(x_t_minus_1 * input_drive)
                recur_drive = gated @ self.W_res.T

            noise = (
                torch.randn_like(x_t) * self.sigma
                if (self.sigma > 0.0 and self.training)
                else 0.0
            )
            x_candidate = input_drive + hist_drive + recur_drive + noise

            if lengths is None:
                x_t = x_candidate
                active_mask = None
            else:
                active_mask = (t < lengths).unsqueeze(1)  # (B, 1), bool
                x_t = torch.where(active_mask, x_candidate, x_t_minus_1)

            vt  = (x_t - x_t_minus_1) * Fs
            A_t = torch.sqrt( x_t**2 + ((vt + self.beta * x_t) / self.omega) ** 2 + 1e-12 )
            if active_mask is None:
                amp_power_accum = amp_power_accum + A_t**2
                x_t_minus_2 = x_t_minus_1
                x_t_minus_1 = x_t
            else:
                amp_power_accum = amp_power_accum + torch.where(active_mask, A_t**2, 0.0)
                x_t_minus_2 = torch.where(active_mask, x_t_minus_1, x_t_minus_2)
                x_t_minus_1 = torch.where(active_mask, x_t, x_t_minus_1)

        if lengths is None:
            features = amp_power_accum / float(T)
        else:
            denom = lengths.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)
            features = amp_power_accum / denom
        features = self.norm(features)
        logits   = self.readout(features)
        return logits

def rrn_extract_features(model: RRNModel, seq: torch.Tensor) -> torch.Tensor:
    """
    seq: shape (T,)  (single MNIST time-series / flattened sequence)
    returns: (K,) features, feature_k = mean_t A_t,k^2
    """
    # Accept numpy or torch, enforce float32 on model device
    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq)
    seq = seq.to(dtype=torch.float32, device=next(model.parameters()).device)

    # Make it (1, T)
    seq_batch = seq.unsqueeze(0)  # (1,T)
    T = seq_batch.shape[1]
    Fs = float(model.Fs)
    dev = seq_batch.device

    # State
    x_t         = torch.zeros(1, model.K, device=dev)
    x_t_minus_1 = torch.zeros(1, model.K, device=dev)
    x_t_minus_2 = torch.zeros(1, model.K, device=dev)
    amp_power_accum = torch.zeros(1, model.K, device=dev)

    w1, w2 = model.w1, model.w2
    W_in   = model.W_in
    beta   = model.beta
    omega  = model.omega

    for t in range(T):
        u_t = seq_batch[:, t]                              # (1,)
        input_drive = u_t.unsqueeze(1) * W_in.unsqueeze(0) # (1,K)
        hist_drive  = w1 * x_t_minus_1 + w2 * x_t_minus_2

        if getattr(model, "no_recurrence", False):
            recur_drive = 0.0
        else:
            gated = torch.tanh(x_t_minus_1 * input_drive)
            recur_drive = gated @ model.W_res.T

        x_t = input_drive + hist_drive + recur_drive

        vt  = (x_t - x_t_minus_1) * Fs
        A_t = torch.sqrt(x_t**2 + ((vt + beta * x_t) / omega) ** 2 + 1e-12)
        amp_power_accum += A_t**2

        x_t_minus_2 = x_t_minus_1
        x_t_minus_1 = x_t

    return (amp_power_accum / float(T)).squeeze(0)  # (K,)
