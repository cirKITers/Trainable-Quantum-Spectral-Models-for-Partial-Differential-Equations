# %%
import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
equation_type = "varpoisson"  # "poisson", "helmholtz", or "varpoisson"
model_kind = "all"  # "spectral_filter_baseline", "diag_phase_free",
# "unitary", "richer_spectral",
# "hhl_like_structured", "hhl_like_free", or "all"

N = 16
max_mode = 6
noise_std = 0  # 0.05

helmholtz_omega = 8.0

varpoisson_alpha = 0.5
varpoisson_eta = 0.0
varpoisson_coeff_type = "linear"

# epsilon = 0 -> diagonal only
# epsilon = 1 -> strongest fixed mixer
richer_epsilon_list = [0.0, 0.25, 0.50, 1.0]

n_hhl_structured_features = 4

n_layers = 3
n_train = 30
n_test = 15
n_epochs = 180
lr = 0.02

seed_list = [0, 1, 2]  # , 3, 4]

n_expr_samples = 120
n_expr_bins = 40
expr_seed = 12345

n_qubits = int(np.log2(N))
assert 2**n_qubits == N

dev_stateprep = qml.device("default.qubit", wires=n_qubits)
dev_ansatz = qml.device("default.qubit", wires=n_qubits)
dev_hhl_like = qml.device("default.qubit", wires=n_qubits + 1)

torch.set_default_dtype(torch.float64)


# ============================================================
# PDE HELPERS
# ============================================================
def dst_matrix(N):
    j = np.arange(1, N + 1)[:, None]
    k = np.arange(1, N + 1)[None, :]
    return np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * j * k / (N + 1))


def laplacian_1d_dirichlet(N, h):
    main = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2


def discrete_laplacian_eigs(N):
    h = 1.0 / (N + 1)
    k = np.arange(1, N + 1)
    return (4.0 / h**2) * np.sin(np.pi * k / (2 * (N + 1))) ** 2


def poisson_matrix(N):
    return laplacian_1d_dirichlet(N, 1.0 / (N + 1))


def helmholtz_matrix(N, omega):
    return poisson_matrix(N) - omega**2 * np.eye(N)


def variable_poisson_matrix(N, alpha, eta=0.0, coeff_type="linear"):
    h = 1.0 / (N + 1)
    x_half = (np.arange(0, N + 1) + 0.5) * h

    if coeff_type == "linear":
        a_half = 1.0 + alpha * x_half
    elif coeff_type == "sin":
        a_half = 1.0 + alpha * np.sin(2.0 * np.pi * x_half)
    else:
        raise ValueError("coeff_type must be 'linear' or 'sin'.")

    if np.min(a_half) <= 0:
        raise ValueError("Coefficient a(x) must remain positive.")

    A = np.zeros((N, N))
    for i in range(N):
        a_imh = a_half[i]
        a_iph = a_half[i + 1]
        A[i, i] = (a_imh + a_iph) / h**2 + eta
        if i > 0:
            A[i, i - 1] = -a_imh / h**2
        if i < N - 1:
            A[i, i + 1] = -a_iph / h**2
    return A


def random_rhs(N, rng, max_mode=6):
    x = np.arange(1, N + 1) / (N + 1)
    coeffs = rng.normal(0.0, 1.0, size=max_mode)
    f = np.zeros(N)
    for k in range(1, max_mode + 1):
        f += coeffs[k - 1] * np.sin(np.pi * k * x)
    return f if np.linalg.norm(f) > 1e-12 else np.eye(1, N, 0).ravel()


def normalize_state(v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def make_dataset(n_samples, equation_type, noise_std, seed, omega=0.0):
    rng = np.random.default_rng(seed)
    data = []

    for _ in range(n_samples):
        f = random_rhs(N, rng, max_mode=max_mode)

        if equation_type == "poisson":
            u_clean = np.linalg.solve(poisson_matrix(N), f)
        elif equation_type == "helmholtz":
            u_clean = np.linalg.solve(helmholtz_matrix(N, omega), f)
        elif equation_type == "varpoisson":
            A = variable_poisson_matrix(
                N,
                alpha=varpoisson_alpha,
                eta=varpoisson_eta,
                coeff_type=varpoisson_coeff_type,
            )
            u_clean = np.linalg.solve(A, f)
        else:
            raise ValueError("Unknown equation_type.")

        u_noisy = u_clean + noise_std * rng.normal(size=u_clean.shape)

        data.append(
            (
                normalize_state(f),
                normalize_state(u_noisy),
                f,
                u_clean,
                u_noisy,
            )
        )

    return data


# ============================================================
# SPECTRAL OBJECTS
# ============================================================
S_np = dst_matrix(N)
S_torch = torch.tensor(S_np, dtype=torch.complex128)

lambda_np = discrete_laplacian_eigs(N)
lambda_torch = torch.tensor(lambda_np, dtype=torch.float64)


# ============================================================
# QNODES
# ============================================================
@qml.qnode(dev_stateprep, interface="torch")
def stateprep_qnode(f_state):
    qml.StatePrep(f_state, wires=range(n_qubits))
    return qml.state()


def apply_hwe_layer(theta_layer):
    for w in range(n_qubits):
        qml.RY(theta_layer[w, 0], wires=w)
    for w in range(n_qubits - 1):
        qml.CNOT(wires=[w, w + 1])

    for w in range(n_qubits):
        qml.RZ(theta_layer[w, 1], wires=w)
    for w in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[w, w + 1])

    for w in range(n_qubits):
        qml.RX(theta_layer[w, 2], wires=w)
    for w in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[w, w + 1])


def apply_hwe_ansatz(theta):
    for l in range(theta.shape[0]):
        apply_hwe_layer(theta[l])


def apply_mixer_bridge(eps):
    """
    eps = 0 -> identity
    eps = 1 -> strongest fixed mixing bridge

    This is a fixed unitary mixer. It introduces non-diagonal mode coupling
    between diagonal phase layers.
    """
    if abs(eps) < 1e-15:
        return

    angle = np.pi * eps / 2.0

    for w in range(n_qubits):
        qml.RY(angle, wires=w)

    for w in range(n_qubits - 1):
        qml.CNOT(wires=[w, w + 1])

    for w in range(n_qubits):
        qml.RZ(angle / 2.0, wires=w)

    for w in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[w - 1, w])

    for w in range(n_qubits):
        qml.RX(angle / 2.0, wires=w)


@qml.qnode(dev_ansatz, interface="torch")
def ansatz_qnode(input_state, theta):
    qml.StatePrep(input_state, wires=range(n_qubits))
    apply_hwe_ansatz(theta)
    return qml.state()


@qml.qnode(dev_ansatz, interface="torch")
def richer_spectral_qnode(input_state, theta, eps):
    """
    Diagonal-richer spectral ansatz.

    theta shape: (n_layers, N)

    eps = 0:
        D_L ... D_1, purely diagonal phase unitary.

    eps = 1:
        D_L B D_{L-1} B ... D_1 with strongest fixed mixer.
    """
    qml.StatePrep(input_state, wires=range(n_qubits))

    for l in range(theta.shape[0]):
        phases = torch.exp(1j * theta[l]).to(torch.complex128)

        qml.DiagonalQubitUnitary(phases, wires=range(n_qubits))

        if l < theta.shape[0] - 1:
            apply_mixer_bridge(float(eps))

    return qml.state()


def int_to_bitlist(k, n_bits):
    return [int(b) for b in format(k, f"0{n_bits}b")]


@qml.qnode(dev_hhl_like, interface="torch")
def hhl_like_qnode(input_state, rotation_angles):
    data_wires = list(range(n_qubits))
    anc = n_qubits

    qml.StatePrep(input_state, wires=data_wires)

    for k in range(N):
        bitvals = int_to_bitlist(k, n_qubits)
        controlled_ry = qml.ctrl(qml.RY, control=data_wires, control_values=bitvals)
        controlled_ry(rotation_angles[k], wires=anc)

    return qml.state()


def hwe_model(f_state, theta):
    return ansatz_qnode(f_state, theta)


# ============================================================
# BASIC HELPERS
# ============================================================
def to_torch_state(x):
    return torch.tensor(x, dtype=torch.float64)


def normalize_torch_state(state, eps=1e-12):
    nrm = torch.linalg.norm(state)
    if torch.abs(nrm) < eps:
        return state
    return state / nrm


def align_global_phase(pred, target):
    overlap = torch.sum(torch.conj(pred) * target)
    if torch.real(overlap) < 0:
        pred = -pred
    return pred


def mse_state_loss(pred, target):
    pred = align_global_phase(pred, target)
    diff = pred - target.to(torch.complex128)
    return torch.real(torch.sum(torch.conj(diff) * diff)) / diff.numel()


def fidelity(pred, target):
    pred = align_global_phase(pred, target)
    overlap = torch.sum(torch.conj(pred) * target.to(torch.complex128))
    return torch.abs(overlap) ** 2


def spectral_coeffs_from_state(state):
    return S_torch @ state


# ============================================================
# GRADIENT STATISTICS
# ============================================================
def sample_loss_model(model_forward, theta, sample):
    f_state, u_state, _, _, _ = sample
    pred = model_forward(to_torch_state(f_state), theta)
    return mse_state_loss(pred, to_torch_state(u_state))


def sample_gradient_statistics(model_forward, theta, dataset):
    grads = []

    for sample in dataset:
        th = theta.clone().detach().requires_grad_(True)
        loss_i = sample_loss_model(model_forward, th, sample)
        grad_i = torch.autograd.grad(
            loss_i, th, retain_graph=False, create_graph=False
        )[0]
        grads.append(grad_i.reshape(-1).detach())

    G = torch.stack(grads, dim=0)
    grad_var_per_param = torch.var(G, dim=0, unbiased=False)

    return {
        "grad_var_mean": torch.mean(grad_var_per_param).item(),
        "grad_var_trace": torch.sum(grad_var_per_param).item(),
        "grad_sq_mean": torch.mean(torch.mean(G**2, dim=0)).item(),
    }


# ============================================================
# EXPRESSIBILITY
# ============================================================
def haar_fidelity_pdf(F, d):
    return (d - 1) * (1 - F) ** (d - 2)


def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))


def random_normalized_state(dim, rng):
    x = rng.normal(size=dim)
    return (x / np.linalg.norm(x)).astype(np.float64)


def fidelity_between_states(psi, phi):
    overlap = torch.sum(torch.conj(psi) * phi)
    return torch.abs(overlap).item() ** 2


def sample_random_theta_like(theta_template, rng, scale=1.0):
    arr = rng.normal(size=tuple(theta_template.shape)) * scale
    return torch.tensor(arr, dtype=torch.float64)


def compute_expressibility(
    model_forward, theta_template, n_samples=120, n_bins=40, seed=12345
):
    rng = np.random.default_rng(seed)
    dim = N
    fidelities = []

    for _ in range(n_samples):
        x1 = torch.tensor(random_normalized_state(dim, rng), dtype=torch.float64)
        x2 = torch.tensor(random_normalized_state(dim, rng), dtype=torch.float64)

        th1 = sample_random_theta_like(theta_template, rng)
        th2 = sample_random_theta_like(theta_template, rng)

        psi1 = normalize_torch_state(model_forward(x1, th1))
        psi2 = normalize_torch_state(model_forward(x2, th2))

        fidelities.append(fidelity_between_states(psi1, psi2))

    fidelities = np.array(fidelities)

    hist_empirical, bin_edges = np.histogram(
        fidelities, bins=n_bins, range=(0.0, 1.0), density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    hist_haar = haar_fidelity_pdf(bin_centers, dim)

    kl_value = kl_divergence(hist_empirical * bin_width, hist_haar * bin_width)

    return kl_value, bin_centers, hist_empirical, hist_haar


# ============================================================
# MODEL FACTORY
# ============================================================
def make_model_forward(model_kind, equation_type, omega=0.0, eps=0.0):
    if equation_type == "poisson":
        den_torch = lambda_torch.clone()
    elif equation_type == "helmholtz":
        den_torch = torch.tensor(lambda_np - omega**2, dtype=torch.float64)
    elif equation_type == "varpoisson":
        den_torch = lambda_torch.clone()
    else:
        raise ValueError("Unknown equation type.")

    def filter_values_from_params(theta):
        a0, a1, a2, a3 = theta
        inv1 = 1.0 / den_torch
        inv2 = 1.0 / den_torch**2
        logterm = 1.0 / torch.log1p(torch.abs(den_torch))
        return a0 + a1 * inv1 + a2 * inv2 + a3 * logterm

    def structured_feature_bank():
        feat0 = torch.ones_like(den_torch)
        inv1 = 1.0 / den_torch
        inv2 = 1.0 / den_torch**2
        feat3 = 1.0 / torch.log1p(torch.abs(den_torch))
        return torch.stack([feat0, inv1, inv2, feat3], dim=0)

    feats = structured_feature_bank()

    def hhl_structured_angles_from_params(theta):
        return torch.sum(theta[:, None] * feats, dim=0)

    def hhl_free_angles_from_params(theta):
        return theta

    def spectral_filter_baseline_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs = S_torch @ state_in
        g = filter_values_from_params(theta).to(torch.complex128)
        coeffs_out = g * coeffs
        return torch.conj(S_torch.T) @ coeffs_out

    def diag_phase_free_model_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs = S_torch @ state_in
        g = torch.exp(1j * theta).to(torch.complex128)
        coeffs_out = g * coeffs
        return torch.conj(S_torch.T) @ coeffs_out

    def spectral_unitary_model_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs_in = S_torch @ state_in
        coeffs_out = ansatz_qnode(coeffs_in, theta)
        return torch.conj(S_torch.T) @ coeffs_out

    def richer_spectral_model_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs_in = S_torch @ state_in
        coeffs_out = richer_spectral_qnode(coeffs_in, theta, eps)
        return torch.conj(S_torch.T) @ coeffs_out

    def hhl_like_structured_model_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs_in = S_torch @ state_in
        angles = hhl_structured_angles_from_params(theta)
        full_state = hhl_like_qnode(coeffs_in, angles).to(torch.complex128)
        full_state = torch.reshape(full_state, (N, 2))
        coeffs_branch = normalize_torch_state(full_state[:, 1])
        return torch.conj(S_torch.T) @ coeffs_branch

    def hhl_like_free_model_forward(f_state, theta):
        state_in = stateprep_qnode(f_state)
        coeffs_in = S_torch @ state_in
        angles = hhl_free_angles_from_params(theta)
        full_state = hhl_like_qnode(coeffs_in, angles).to(torch.complex128)
        full_state = torch.reshape(full_state, (N, 2))
        coeffs_branch = normalize_torch_state(full_state[:, 1])
        return torch.conj(S_torch.T) @ coeffs_branch

    if model_kind == "spectral_filter_baseline":
        return spectral_filter_baseline_forward, filter_values_from_params
    if model_kind == "diag_phase_free":
        return diag_phase_free_model_forward, lambda theta: theta
    if model_kind == "unitary":
        return spectral_unitary_model_forward, None
    if model_kind == "richer_spectral":
        return richer_spectral_model_forward, None
    if model_kind == "hhl_like_structured":
        return hhl_like_structured_model_forward, hhl_structured_angles_from_params
    if model_kind == "hhl_like_free":
        return hhl_like_free_model_forward, hhl_free_angles_from_params

    raise ValueError("Unknown model_kind.")


# ============================================================
# METRICS
# ============================================================
def dataset_loss_model(model_forward, theta, dataset):
    return torch.stack(
        [
            mse_state_loss(
                model_forward(to_torch_state(f_state), theta),
                to_torch_state(u_state),
            )
            for f_state, u_state, _, _, _ in dataset
        ]
    ).mean()


def dataset_fidelity_model(model_forward, theta, dataset):
    vals = []
    for f_state, u_state, _, _, _ in dataset:
        pred = model_forward(to_torch_state(f_state), theta)
        vals.append(fidelity(pred, to_torch_state(u_state)))
    return torch.stack(vals).mean().item()


def target_spectrum(dataset):
    vals = []
    for _, u_state, _, _, _ in dataset:
        u_t = to_torch_state(u_state).to(torch.complex128)
        c_true = spectral_coeffs_from_state(u_t)
        vals.append(torch.real(torch.abs(c_true) ** 2))
    return torch.stack(vals).mean(dim=0).detach().cpu().numpy()


def modewise_abs_error_model(model_forward, theta, dataset):
    errs = []
    for f_state, u_state, _, _, _ in dataset:
        pred = model_forward(to_torch_state(f_state), theta)
        u_t = to_torch_state(u_state).to(torch.complex128)
        pred = align_global_phase(pred, u_t)
        c_pred = spectral_coeffs_from_state(pred)
        c_true = spectral_coeffs_from_state(u_t)
        errs.append(torch.real(torch.abs(c_pred - c_true) ** 2))
    return torch.stack(errs).mean(dim=0).detach().cpu().numpy()


def relative_error(abs_err, target_spec, eps=1e-12):
    return abs_err / (target_spec + eps)


def mode_gradient_power_all_params(model_forward_fn, theta, dataset):
    n_params = theta.numel()
    powers = []

    for f_state, _, _, _, _ in dataset:
        th = theta.clone().detach().requires_grad_(True)
        state = model_forward_fn(to_torch_state(f_state), th)
        coeffs = S_torch @ state

        mode_vals = []
        for k in range(N):
            ck = torch.real(coeffs[k])
            grad_ck = torch.autograd.grad(
                ck, th, retain_graph=True, create_graph=False
            )[0]
            mode_vals.append((torch.sum(grad_ck**2) / n_params).detach())

        powers.append(torch.stack(mode_vals))

    return torch.stack(powers).mean(dim=0).detach().cpu().numpy()


# ============================================================
# INITIALIZATION
# ============================================================
def init_filter_theta():
    return torch.tensor([0.0, 0.2, 0.0, 0.0], dtype=torch.float64, requires_grad=True)


def init_unitary_theta(seed, scale=0.1):
    rng = np.random.default_rng(seed)
    arr = scale * rng.normal(size=(n_layers, n_qubits, 3))
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def init_diag_phase_free_theta(seed, scale=0.1):
    rng = np.random.default_rng(seed)
    arr = scale * rng.normal(size=(N,))
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def init_richer_diag_theta(seed, scale=0.1):
    rng = np.random.default_rng(seed)
    arr = scale * rng.normal(size=(n_layers, N))
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def init_hhl_structured_theta(seed, scale=0.03):
    rng = np.random.default_rng(seed)
    arr = scale * rng.normal(size=(n_hhl_structured_features,))
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def init_hhl_free_theta(seed, scale=0.03):
    rng = np.random.default_rng(seed)
    arr = scale * rng.normal(size=(N,))
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


# ============================================================
# RUN ONE SEED
# ============================================================
def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = make_dataset(
        n_train, equation_type, noise_std, seed, omega=helmholtz_omega
    )
    test_data = make_dataset(
        n_test, equation_type, noise_std, seed + 100, omega=helmholtz_omega
    )

    if model_kind == "all":
        model_specs = [
            ("spectral_filter_baseline", None),
            ("diag_phase_free", None),
            ("unitary", None),
            *[("richer_spectral", eps) for eps in richer_epsilon_list],
            ("hhl_like_structured", None),
            ("hhl_like_free", None),
        ]
    elif model_kind == "richer_spectral":
        model_specs = [("richer_spectral", eps) for eps in richer_epsilon_list]
    else:
        model_specs = [(model_kind, None)]

    results = {}

    for idx, (mk, eps_val) in enumerate(model_specs):
        forward, aux_fun = make_model_forward(
            mk,
            equation_type,
            omega=helmholtz_omega,
            eps=0.0 if eps_val is None else eps_val,
        )

        if mk == "spectral_filter_baseline":
            theta = init_filter_theta()
            model_key = "spectral_filter_baseline"
        elif mk == "diag_phase_free":
            theta = init_diag_phase_free_theta(seed + 1100)
            model_key = "diag_phase_free"
        elif mk == "unitary":
            theta = init_unitary_theta(seed + 1300)
            model_key = "unitary"
        elif mk == "richer_spectral":
            theta = init_richer_diag_theta(seed + 1400 + idx)
            model_key = f"richer_spectral_eps_{eps_val:.2f}"
        elif mk == "hhl_like_structured":
            theta = init_hhl_structured_theta(seed + 1500)
            model_key = "hhl_like_structured"
        elif mk == "hhl_like_free":
            theta = init_hhl_free_theta(seed + 1600)
            model_key = "hhl_like_free"

        opt = torch.optim.Adam([theta], lr=lr)

        train_loss, test_loss, train_fid, test_fid = [], [], [], []

        for _ in range(n_epochs):
            opt.zero_grad()
            loss = dataset_loss_model(forward, theta, train_data)
            loss.backward()
            opt.step()

            with torch.no_grad():
                train_loss.append(dataset_loss_model(forward, theta, train_data).item())
                test_loss.append(dataset_loss_model(forward, theta, test_data).item())
                train_fid.append(dataset_fidelity_model(forward, theta, train_data))
                test_fid.append(dataset_fidelity_model(forward, theta, test_data))

        target_spec = target_spectrum(test_data)
        abs_err = modewise_abs_error_model(forward, theta, test_data)
        rel_err = relative_error(abs_err, target_spec)
        grad_power = mode_gradient_power_all_params(forward, theta, test_data)
        grad_stats = sample_gradient_statistics(forward, theta, test_data)

        expr_val, expr_bins, expr_hist, expr_haar = compute_expressibility(
            forward,
            theta.detach(),
            n_samples=n_expr_samples,
            n_bins=n_expr_bins,
            seed=expr_seed + seed + 100 * idx,
        )

        results[model_key] = {
            "train_loss": np.array(train_loss),
            "test_loss": np.array(test_loss),
            "train_fid": np.array(train_fid),
            "test_fid": np.array(test_fid),
            "target_spec": target_spec,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "grad_power": grad_power,
            "expressibility": expr_val,
            "expr_bins": expr_bins,
            "expr_hist": expr_hist,
            "expr_haar": expr_haar,
            "grad_var_mean": grad_stats["grad_var_mean"],
            "grad_var_trace": grad_stats["grad_var_trace"],
            "grad_sq_mean": grad_stats["grad_sq_mean"],
            "theta_numel": theta.numel(),
        }

    # HWE baseline
    theta_hwe = init_unitary_theta(seed + 9999)
    opt_hwe = torch.optim.Adam([theta_hwe], lr=lr)

    train_loss, test_loss, train_fid, test_fid = [], [], [], []

    for _ in range(n_epochs):
        opt_hwe.zero_grad()
        loss = dataset_loss_model(hwe_model, theta_hwe, train_data)
        loss.backward()
        opt_hwe.step()

        with torch.no_grad():
            train_loss.append(
                dataset_loss_model(hwe_model, theta_hwe, train_data).item()
            )
            test_loss.append(dataset_loss_model(hwe_model, theta_hwe, test_data).item())
            train_fid.append(dataset_fidelity_model(hwe_model, theta_hwe, train_data))
            test_fid.append(dataset_fidelity_model(hwe_model, theta_hwe, test_data))

    target_spec = target_spectrum(test_data)
    abs_err = modewise_abs_error_model(hwe_model, theta_hwe, test_data)
    rel_err = relative_error(abs_err, target_spec)
    grad_power = mode_gradient_power_all_params(hwe_model, theta_hwe, test_data)
    grad_stats = sample_gradient_statistics(hwe_model, theta_hwe, test_data)

    expr_val, expr_bins, expr_hist, expr_haar = compute_expressibility(
        hwe_model,
        theta_hwe.detach(),
        n_samples=n_expr_samples,
        n_bins=n_expr_bins,
        seed=expr_seed + 5000 + seed,
    )

    results["hwe"] = {
        "train_loss": np.array(train_loss),
        "test_loss": np.array(test_loss),
        "train_fid": np.array(train_fid),
        "test_fid": np.array(test_fid),
        "target_spec": target_spec,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "grad_power": grad_power,
        "expressibility": expr_val,
        "expr_bins": expr_bins,
        "expr_hist": expr_hist,
        "expr_haar": expr_haar,
        "grad_var_mean": grad_stats["grad_var_mean"],
        "grad_var_trace": grad_stats["grad_var_trace"],
        "grad_sq_mean": grad_stats["grad_sq_mean"],
        "theta_numel": theta_hwe.numel(),
    }

    return results


# ============================================================
# RUN ALL SEEDS
# ============================================================
all_results = []

for seed in seed_list:
    print(f"Running seed {seed}")
    all_results.append(run_one_seed(seed))


# ============================================================
# AGGREGATION
# ============================================================
all_model_keys = list(all_results[0].keys())


def mean_std_model(model_key, metric_key):
    arr = np.stack([r[model_key][metric_key] for r in all_results], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


aggregated = {}

for mk in all_model_keys:
    aggregated[mk] = {}
    for key in [
        "train_loss",
        "test_loss",
        "train_fid",
        "test_fid",
        "target_spec",
        "abs_err",
        "rel_err",
        "grad_power",
        "expressibility",
        "expr_bins",
        "expr_hist",
        "expr_haar",
        "grad_var_mean",
        "grad_var_trace",
        "grad_sq_mean",
    ]:
        aggregated[mk][key + "_mean"], aggregated[mk][key + "_std"] = mean_std_model(
            mk, key
        )

    aggregated[mk]["theta_numel"] = all_results[0][mk]["theta_numel"]


# ============================================================
# PLOTS NO PAPER-READY BUT USEFUL FOR ANALYSIS
# ============================================================
title_suffix = equation_type.capitalize()
if equation_type == "helmholtz":
    title_suffix += f" (omega={helmholtz_omega})"
elif equation_type == "varpoisson":
    title_suffix += f" (alpha={varpoisson_alpha}, eta={varpoisson_eta}, coeff={varpoisson_coeff_type})"
title_suffix += f", noise={noise_std}"

label_map = {
    "spectral_filter_baseline": "Spectral filter baseline",
    "diag_phase_free": "Diagonal phase free",
    "unitary": "Spectral unitary",
    "hhl_like_structured": "HHL-like structured",
    "hhl_like_free": "HHL-like free",
    "hwe": "HWE",
}

for eps in richer_epsilon_list:
    label_map[f"richer_spectral_eps_{eps:.2f}"] = (
        rf"Diag-richer spectral ($\epsilon={eps:.2f}$)"
    )

base_markers = ["o", "x", "^", "d", "p", "h", "s", "*", "v"]
style_map = {}

for i, mk in enumerate(all_model_keys):
    marker = base_markers[i % len(base_markers)]
    style_map[mk] = {"train": marker + "-", "test": marker + "--"}

epochs = np.arange(n_epochs)
k_plot = np.arange(1, N + 1)

plt.figure(figsize=(9, 5.2))
for mk in all_model_keys:
    plt.plot(
        epochs,
        aggregated[mk]["train_loss_mean"],
        style_map[mk]["train"],
        label=f"{label_map[mk]} train",
    )
    plt.plot(
        epochs,
        aggregated[mk]["test_loss_mean"],
        style_map[mk]["test"],
        label=f"{label_map[mk]} test",
    )
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title(f"{title_suffix} - Loss")
plt.grid(True)
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()

plt.figure(figsize=(9, 5.2))
for mk in all_model_keys:
    plt.plot(
        epochs,
        aggregated[mk]["train_fid_mean"],
        style_map[mk]["train"],
        label=f"{label_map[mk]} train",
    )
    plt.plot(
        epochs,
        aggregated[mk]["test_fid_mean"],
        style_map[mk]["test"],
        label=f"{label_map[mk]} test",
    )
plt.xlabel("Epoch")
plt.ylabel("Mean fidelity")
plt.title(f"{title_suffix} - Fidelity")
plt.grid(True)
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()

plt.figure(figsize=(9, 5.2))
for mk in all_model_keys:
    plt.plot(
        k_plot,
        aggregated[mk]["abs_err_mean"],
        style_map[mk]["train"],
        label=label_map[mk],
    )
plt.xlabel("Mode index k")
plt.ylabel("Absolute spectral error")
plt.title(f"{title_suffix} - Absolute spectral error")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()

plt.figure(figsize=(9, 5.2))
for mk in all_model_keys:
    plt.plot(
        k_plot,
        aggregated[mk]["rel_err_mean"],
        style_map[mk]["train"],
        label=label_map[mk],
    )
plt.xlabel("Mode index k")
plt.ylabel("Relative spectral error")
plt.title(f"{title_suffix} - Relative spectral error")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()

plt.figure(figsize=(9, 5.2))
for mk in all_model_keys:
    plt.plot(
        k_plot,
        aggregated[mk]["grad_power_mean"],
        style_map[mk]["train"],
        label=label_map[mk],
    )
plt.xlabel("Mode index k")
plt.ylabel(r"$\frac{1}{P}\sum_p \mathbb{E}[|\partial_{\theta_p} c_k|^2]$")
plt.title(f"{title_suffix} - Gradient power per mode")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()

plt.figure(figsize=(10, 4.8))
labels = [label_map[mk] for mk in all_model_keys]
means = [aggregated[mk]["expressibility_mean"] for mk in all_model_keys]
stds = [aggregated[mk]["expressibility_std"] for mk in all_model_keys]
x = np.arange(len(labels))
plt.bar(x, means, yerr=stds, alpha=0.85, capsize=5)
plt.xticks(x, labels, rotation=18, ha="right")
plt.ylabel("KL to Haar")
plt.title(f"{title_suffix} - Expressibility metric (lower = more expressive)")
plt.grid(True, axis="y")
plt.tight_layout()

plt.figure(figsize=(10, 4.8))
gv_means = [aggregated[mk]["grad_var_mean_mean"] for mk in all_model_keys]
gv_stds = [aggregated[mk]["grad_var_mean_std"] for mk in all_model_keys]
x = np.arange(len(all_model_keys))
plt.bar(x, gv_means, yerr=gv_stds, alpha=0.85, capsize=5)
plt.xticks(x, [label_map[mk] for mk in all_model_keys], rotation=18, ha="right")
plt.ylabel("Mean parameter-wise gradient variance")
plt.title(f"{title_suffix} - Gradient variance")
plt.grid(True, axis="y")
plt.tight_layout()

plt.show()


# ============================================================
# SUMMARY
# ============================================================
print("\nFinal metrics over seeds")
print("-" * 110)
print(f"Equation type       : {equation_type}")
print(f"Model kind          : {model_kind}")
print(f"richer epsilons     : {richer_epsilon_list}")
print(f"noise std           : {noise_std}")
print(f"n_qubits            : {n_qubits}")
print(f"n_layers            : {n_layers}")

for mk in all_model_keys:
    print("-" * 110)
    print(f"{label_map[mk]}")
    print(f"#params                 : {aggregated[mk]['theta_numel']}")
    print(
        f"Test loss               : {aggregated[mk]['test_loss_mean'][-1]:.6f} ± {aggregated[mk]['test_loss_std'][-1]:.6f}"
    )
    print(
        f"Test fidelity           : {aggregated[mk]['test_fid_mean'][-1]:.6f} ± {aggregated[mk]['test_fid_std'][-1]:.6f}"
    )
    print(
        f"KL to Haar              : {aggregated[mk]['expressibility_mean']:.6f} ± {aggregated[mk]['expressibility_std']:.6f}"
    )
    print(
        f"Grad variance mean      : {aggregated[mk]['grad_var_mean_mean']:.6e} ± {aggregated[mk]['grad_var_mean_std']:.6e}"
    )
    print(
        f"Grad variance trace     : {aggregated[mk]['grad_var_trace_mean']:.6e} ± {aggregated[mk]['grad_var_trace_std']:.6e}"
    )
    print(
        f"Mean squared gradient   : {aggregated[mk]['grad_sq_mean_mean']:.6e} ± {aggregated[mk]['grad_sq_mean_std']:.6e}"
    )

print("Spectral filter is included only as a classical baseline.")
print(
    "Diag-richer spectral: epsilon=0 is diagonal; epsilon=1 uses strongest fixed unitary mixer."
)

# %%
# ============================================================
# SAVE RESULTS FOR PLOTTING
# ============================================================
import pickle
import os
from datetime import datetime

results_dir = "Data/VarPoisson"

os.makedirs(results_dir, exist_ok=True)

run_name = (
    f"{equation_type}_N{N}_epochs{n_epochs}_"
    f"seeds{len(seed_list)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

results_path = os.path.join(results_dir, f"{run_name}.pkl")

payload = {
    # main objects needed by plotting scripts
    "all_results": all_results,
    "aggregated": aggregated,
    "all_model_keys": all_model_keys,
    # variables expected by plotting scripts
    "N": N,
    "n_epochs": n_epochs,
    "richer_epsilon_list": richer_epsilon_list,
    # useful metadata
    "equation_type": equation_type,
    "model_kind": model_kind,
    "noise_std": noise_std,
    "helmholtz_omega": helmholtz_omega,
    "varpoisson_alpha": varpoisson_alpha,
    "varpoisson_eta": varpoisson_eta,
    "varpoisson_coeff_type": varpoisson_coeff_type,
    "n_layers": n_layers,
    "n_train": n_train,
    "n_test": n_test,
    "lr": lr,
    "seed_list": seed_list,
    "n_expr_samples": n_expr_samples,
    "n_expr_bins": n_expr_bins,
    "expr_seed": expr_seed,
}

with open(results_path, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved results to:\n{results_path}")
