#!/usr/bin/env python3
"""Self-contained PRDM numerical experiments and PCM-style plot.

Run:
    python prdm_all.py --mode all --shots 100000 --trials 20
    python prdm_all.py --mode plot
    python prdm_all.py --mode all --input saved --shots 100000 --trials 20

Default input is |Psi>=(|00>+|11>+|22>+|33>)/2 on two 12-qubit parties.
Set --input saved to read Schmidt_lambdas.npy / Schmidt_vectors_A.npy.
MUBs: S=2 or odd prime (not general prime powers).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S_LIST = (5, 7, 11, 17, 23, 31, 41, 47)
D_DEFAULT = 4096
ETA = 0.01
KAPPA = 0.5
EPS = 0.02
LAM_TOL = 1e-12
FIGSIZE = (8.5, 12.0)
DPI = 600

# ---------------------------- PRDM numerical helpers ----------------------------
def is_prime(n: int) -> bool:
    return n >= 2 and all(n % j for j in range(2, int(n**0.5) + 1))


def complete_mubs(S: int) -> list[np.ndarray]:
    """Return S+1 bases, with basis vectors in columns."""
    if S == 2:
        return [np.eye(2, dtype=complex),
                np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),
                np.array([[1, 1], [1j, -1j]], complex) / np.sqrt(2)]
    if S % 2 == 0 or not is_prime(S):
        raise ValueError('Complete MUBs require S=2 or an odd prime.')
    x = np.arange(S)[:, None]
    k = np.arange(S)[None, :]
    bases = [np.eye(S, dtype=complex)]
    for a in range(S):
        phase = (a*x*x + x*k) % S
        bases.append(np.exp(2j*np.pi*phase/S) / np.sqrt(S))
    return bases


def validate_spectrum(lam) -> np.ndarray:
    lam = np.asarray(lam, float)
    if lam.ndim != 1 or not len(lam) or np.any(lam <= 0) or not np.isfinite(lam).all():
        raise ValueError('lam must contain strictly positive eigenvalues.')
    if not np.isclose(lam.sum(), 1, atol=1e-8):
        raise ValueError('lam must sum to 1.')
    return lam


def validate_vectors(V: np.ndarray, lam: np.ndarray) -> np.ndarray:
    V = np.asarray(V, complex)
    if V.ndim != 2 or V.shape[1] != len(lam) or V.shape[0] < len(lam):
        raise ValueError('V must have shape (d, rank).')
    if not np.allclose(V.conj().T @ V, np.eye(len(lam)), atol=1e-7):
        raise ValueError('Columns of V must be orthonormal.')
    return V


def haar_isometry(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform d-by-r Stiefel isometry via complex Gaussian QR."""
    if not 1 <= r <= d:
        raise ValueError('Require 1 <= r <= d.')
    z = rng.normal(size=(d, r)) + 1j*rng.normal(size=(d, r))
    q, rr = np.linalg.qr(z, mode='reduced')
    diag = np.diag(rr)
    return q * (diag / np.abs(diag))[None, :]


def projected_density(B: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """B is the first S rows of the rotated d-by-r eigenvector matrix."""
    return (B * lam[None, :]) @ B.conj().T


def reconstruct_mubs(omega: np.ndarray, shots: int, bases: list[np.ndarray], rng):
    """Simulate all S+1 MUB settings, including projection failures."""
    S = omega.shape[0]
    if not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError('shots must be a positive integer')
    if len(bases) != S + 1:
        raise ValueError('Expected S+1 MUB bases.')
    p = float(np.trace(omega).real)
    if p < -1e-9 or p > 1 + 1e-9:
        raise ValueError('Projection probability out of range')
    freq_matrix = np.zeros((S, S), complex)
    n_success = 0
    for basis in bases:
        prob = np.real(np.diag(basis.conj().T @ omega @ basis))
        prob = np.maximum(prob, 0)
        probs = np.r_[prob, max(0.0, 1-p)]
        probs /= probs.sum()
        counts = rng.multinomial(shots, probs)
        n_success += int(counts[:-1].sum())
        freq_matrix += (basis * (counts[:-1] / shots)[None, :]) @ basis.conj().T
    phat = n_success / ((S+1) * shots)
    estimate = freq_matrix - phat * np.eye(S)
    return (estimate + estimate.conj().T) / 2, phat


def signal_lower_bound(S, d, lam_lb, kappa):
    return (1-kappa) * (S/d) * lam_lb


def noise_threshold(S, d, shots, eta, kappa, *, conservative=False):
    """Noise cutoff; eta/2 is allocated to tomography."""
    if shots < 1 or not 0 < eta < 1 or not 0 < kappa < 1:
        raise ValueError('Invalid shots, eta, or kappa.')
    L = np.log(4*S/eta)
    pstar = 1.0 if conservative else min(1.0, (1+kappa)*S/d)
    return np.sqrt(4*pstar*L/shots) + 2*L/(3*shots)


def sufficient_shots(S, d, lam_lb, kappa, eta):
    kS = signal_lower_bound(S, d, lam_lb, kappa)
    if kS <= 0:
        raise ValueError('Signal lower bound must be positive.')
    pstar = min(1.0, (1+kappa)*S/d)
    return int(np.ceil((64*pstar/kS**2 + 8/(3*kS))*np.log(4*S/eta)))


def rank_estimate(estimate: np.ndarray, threshold: float):
    """Count eigenvalues strictly larger than the noise threshold."""
    eig = np.linalg.eigvalsh(estimate)[::-1]
    return int(np.count_nonzero(eig > threshold)), eig


def run_trial(B, lam, d, shots, eta, kappa, bases, rng, *, conservative=False):
    """Return rank, estimated/true eigenvalues, cutoff, and projected weight."""
    S, r = B.shape
    omega = projected_density(B, lam)
    estimate, phat = reconstruct_mubs(omega, shots, bases, rng)
    threshold = noise_threshold(S, d, shots, eta, kappa, conservative=conservative)
    rank_hat, eig_hat = rank_estimate(estimate, threshold)
    return rank_hat, eig_hat, np.linalg.eigvalsh(omega)[::-1], threshold, phat


# ----------------------- Data, simulation and diagnostics -----------------------

def load_state(data_dir: Path, choice: str, dimension: int):
    if choice == 'structured':
        if dimension < 4:
            raise ValueError('dimension must be >= 4')
        lam = np.full(4, 0.25)
        V0 = np.eye(dimension, 4, dtype=complex)
        return lam, V0
    lam_all = np.asarray(np.load(data_dir / 'Schmidt_lambdas.npy'), dtype=float).ravel()
    if not np.isfinite(lam_all).all() or lam_all.sum() <= 0:
        raise ValueError('Invalid saved Schmidt probabilities.')
    lam_all /= lam_all.sum()
    keep = lam_all > LAM_TOL
    if not np.isclose(lam_all[~keep].sum(), 0, atol=1e-10):
        raise ValueError('Nontrivial Schmidt weight was truncated.')
    lam = validate_spectrum(lam_all[keep] / lam_all[keep].sum())
    raw = np.load(data_dir / 'Schmidt_vectors_A.npy')
    if raw.shape[1] != len(lam_all):
        raise ValueError('Schmidt vectors and probabilities do not align.')
    return lam, validate_vectors(raw[:, keep], lam)


def projector_indices(S, d, r, mode):
    # Same fixed projector for rotated and unrotated comparisons.
    if mode == 'rank_one':
        if r != 4 or S > d-3:
            raise ValueError('rank_one projector requires chi=4, S<=d-3.')
        indices = np.r_[0, np.arange(4, S+3)]
    else:
        indices = np.arange(S)
    if len(indices) != S or len(np.unique(indices)) != S:
        raise AssertionError('Invalid projector coordinate set.')
    return indices


def save_csv(path, header, rows):
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=',',
               header=header, comments='')
    print(f'Saved {path}')


def simulate_rotated(lam, V0, S_values, shots, trials, eta, kappa, eps,
                     projector, output):
    d, r = V0.shape
    rng = np.random.default_rng(42)
    rows = []
    print('Rotated PRDM: S  exact  MUB   noisy  threshold  max_tail')
    for S in S_values:
        idx = projector_indices(S, d, r, projector)
        bases = complete_mubs(S)
        cutoff = noise_threshold(S, d, shots, eta, kappa)
        counts = np.zeros((3, trials))
        tails = np.full(trials, np.nan)
        for t in range(trials):
            V = haar_isometry(d, r, rng)
            omega = projected_density(V[idx, :], lam)
            rho_hat, _ = reconstruct_mubs(omega, shots, bases, rng)
            omega_noisy = (1-eps)*omega + (eps/d)*np.eye(S)
            noisy_hat, _ = reconstruct_mubs(omega_noisy, shots, bases, rng)
            counts[0,t] = np.count_nonzero(np.linalg.eigvalsh(omega)>cutoff)
            counts[1,t] = rank_estimate(rho_hat, cutoff)[0]
            counts[2,t] = rank_estimate(noisy_hat, cutoff)[0]
            if S>r:
                tails[t] = np.linalg.eigvalsh(rho_hat)[-(r+1)]
        row = [S, *counts.mean(axis=1),
               np.mean(counts[1]>=r), np.mean(counts[2]>=r), cutoff,
               np.nanmean(tails) if S>r else np.nan,
               np.nanmax(tails) if S>r else np.nan,
               *counts.std(axis=1)]
        rows.append(row)
        print(f'{S:3d} {row[1]:6.2f} {row[2]:6.2f} {row[3]:6.2f}'
              f' {cutoff:.4g} {row[9]:.4g}')
    save_csv(output,
             'S,exact_count,mub_count,noisy_count,P_cert,P_noisy,threshold,'
             'mean_lambda_chi_plus_1,max_lambda_chi_plus_1,'
             'exact_count_std,mub_count_std,noisy_count_std', rows)


def simulate_unrotated(lam, V0, S_values, shots, trials, eta, kappa,
                       projector, output):
    d, r = V0.shape
    rng = np.random.default_rng(43)
    null_lam = lam[:-1]/lam[:-1].sum() if r>1 else None
    rows=[]
    print('Unrotated PRDM: S exact_rank exact_stable mean_count threshold')
    for S in S_values:
        idx=projector_indices(S,d,r,projector)
        B=V0[idx,:]
        omega=projected_density(B,lam)
        eig=np.linalg.eigvalsh(omega)
        exact_rank=int(np.linalg.matrix_rank(B))
        if projector=='rank_one' and exact_rank!=1:
            raise ValueError(f'Expected exact projected rank 1, got {exact_rank} for S={S}. '
                             'The saved input state is not the structured basis state.')
        # Conservative p*=1 is necessary when there is no Haar rotation.
        cutoff=noise_threshold(S,d,shots,eta,kappa,conservative=True)
        exact_stable=int(np.count_nonzero(eig>cutoff))
        bases=complete_mubs(S)
        omega_null=projected_density(B[:,:-1],null_lam) if r>1 else None
        counts=np.zeros((2,trials))
        for t in range(trials):
            estimated,_=reconstruct_mubs(omega,shots,bases,rng)
            counts[0,t]=rank_estimate(estimated,cutoff)[0]
            if r>1:
                null_hat,_=reconstruct_mubs(omega_null,shots,bases,rng)
                counts[1,t]=rank_estimate(null_hat,cutoff)[0]
        row=[S, np.mean(counts[0]>=r),np.mean(counts[1]>=r),
             counts[0].mean(),counts[1].mean(),cutoff,exact_rank,exact_stable,
             counts[0].std(),counts[1].std()]
        rows.append(row)
        print(f'{S:3d} {exact_rank:3d} {exact_stable:3d} {row[3]:7.2f} {cutoff:.4g}')
    save_csv(output,
             'S,P_cert,P_false,mean_count,null_count,threshold,exact_rank,'
             'exact_stable_rank,mean_count_std,null_count_std',rows)


def load_csv(filename, required):
    data=np.atleast_1d(np.genfromtxt(filename,delimiter=',',names=True))
    if data.size==0 or data.dtype.names is None:
        raise ValueError(f'Empty or invalid CSV: {filename}')
    missing=[name for name in required if name not in data.dtype.names]
    if missing:
        raise ValueError(f'{filename}: missing columns {missing}')
    return data[np.argsort(data['S'])]


# ------------------------------ PCM-matched plotting ------------------------------

def configure_matplotlib():
    have_latex = shutil.which('latex') is not None and shutil.which('dvipng') is not None
    if have_latex:
        try:
            result = subprocess.run(['kpsewhich', 'type1ec.sty'], capture_output=True,
                                    text=True, timeout=10, check=False)
            have_latex = bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            have_latex = False
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 18,
        'axes.labelsize': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'text.usetex': have_latex,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{times}\usepackage{amsmath,amssymb}'
        )


def bold_int_labels(values):
    return [rf'$\boldsymbol{{{int(round(v))}}}$' for v in values]


def nice_step(span, target=6):
    if span <= 0:
        return 1.0
    raw = span / target
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10.0 * magnitude


def bold_decimal_ticks(ax, ymax):
    step = nice_step(ymax)
    ticks = np.arange(0, ymax + 0.25 * step, step)
    ticks = ticks[ticks <= ymax + 1e-10]
    decimals = max(0, min(8, int(np.ceil(-np.log10(step))) + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf'$\boldsymbol{{{v:.{decimals}f}}}$' for v in ticks])


def std_column(data, key):
    # Error bars are drawn only if the simulation actually saved trial std.
    for candidate in (key + '_std', 'std_' + key):
        if candidate in data.dtype.names:
            return np.maximum(0.0, np.asarray(data[candidate], dtype=float))
    return np.zeros(len(data), dtype=float)





def draw_plot(rotated_path, unrotated_path, chi, output, first_half=False,
              use_max_tail=True):
    rotated=load_csv(rotated_path, ('S','threshold','exact_count','mub_count',
        'noisy_count','mean_lambda_chi_plus_1','max_lambda_chi_plus_1'))
    unrotated=load_csv(unrotated_path,('S','exact_stable_rank','exact_rank'))
    if first_half:
        rotated=rotated[:max(1,len(rotated)//2)]
        unrotated=unrotated[:max(1,len(unrotated)//2)]
    Sr,Su=rotated['S'],unrotated['S']
    threshold=np.asarray(rotated['threshold'],float)
    scale=float(np.max(threshold))
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid threshold normalization.')
    tail_key='max_lambda_chi_plus_1' if use_max_tail else 'mean_lambda_chi_plus_1'
    tail=rotated[tail_key]/scale
    normal=threshold/scale
    finite=np.isfinite(tail)
    configure_matplotlib()
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=FIGSIZE)
    ax1.set_title(r'$\boldsymbol{\mathrm{(a)}}$')
    ax1.plot(Sr,normal,'--s',color='black',linewidth=2,markersize=6,
       label=r'$\boldsymbol{\lambda_{\mathrm{th}}/\lambda_{\mathrm{th},\max}}$')
    tail_label=(r'$\boldsymbol{\max_{\mathrm{trials}}\lambda_{\chi+1}'
       r'(\widehat{\rho}_{U,S})/\lambda_{\mathrm{th},\max}}$' if use_max_tail else
       r'$\boldsymbol{\langle\lambda_{\chi+1}(\widehat{\rho}_{U,S})'
       r'\rangle/\lambda_{\mathrm{th},\max}}$')
    ax1.plot(Sr[finite],tail[finite],'-v',color='crimson',linewidth=1.8,
             markersize=7,label=tail_label)
    ax1.fill_between(Sr[finite],tail[finite],normal[finite],alpha=.18,color='gray')
    ax1.set_ylabel(r'$\boldsymbol{\mathrm{Normalized\ eigenvalue}}$')
    ylo=min(0.,float(np.min(tail[finite]))*1.15) if finite.any() else 0.
    ymax=1.15*max(1.,float(np.max(tail[finite])) if finite.any() else 1.)
    ax1.set_ylim(ylo,ymax)
    ax1.set_xticks(Sr)
    ax1.set_xticklabels(bold_int_labels(Sr))
    if ylo>=0: bold_decimal_ticks(ax1,ymax)
    ax1.legend(loc='upper left')

    ax2.set_title(r'$\boldsymbol{\mathrm{(b)}}$')
    def curve(x,values,std,color,ls,marker,label):
        ax2.errorbar(x,values,yerr=std,color=color,linestyle=ls,marker=marker,
                     markersize=6,linewidth=1.9,elinewidth=1.3,capsize=4,
                     capthick=1.2,label=label)
    def std(data,key):
        col=key+'_std'
        return np.asarray(data[col]) if col in data.dtype.names else np.zeros(len(data))
    curve(Sr,rotated['exact_count'],std(rotated,'exact_count'),'royalblue','-','o',
          r'$\boldsymbol{\rho_{U,S},\ \mathrm{exact}}$')
    curve(Sr,rotated['mub_count'],std(rotated,'mub_count'),'red','-','^',
          r'$\boldsymbol{\widehat{\rho}_{U,S},\ \mathrm{MUB}}$')
    # curve(Sr,rotated['noisy_count'],std(rotated,'noisy_count'),'goldenrod','-','d',
    #       r'$\boldsymbol{\widehat{\rho}^{\mathcal E}_{U,S},\ \mathrm{MUB}}$')
    curve(Su,unrotated['exact_stable_rank'],np.zeros(len(Su)),'seagreen','--','s',
          r'$\boldsymbol{\rho_S,\ \mathrm{exact}}$')
    ax2.axhline(chi-1,color='black',linestyle='--',linewidth=1.8,
                label=rf'$\boldsymbol{{\chi={chi-1}}}$')
    xt=np.unique(np.r_[Sr,Su])
    ax2.set_xticks(xt)
    ax2.set_xticklabels(bold_int_labels(xt))
    ymax=max(chi+1,int(np.ceil(max(np.max(rotated['exact_count']),
         np.max(rotated['mub_count']),np.max(rotated['noisy_count']),
         np.max(unrotated['exact_stable_rank']))+1)))
    ax2.set_ylim(0,ymax)
    yt=np.arange(0,ymax+1)
    ax2.set_yticks(yt)
    ax2.set_yticklabels(bold_int_labels(yt))
    ax2.set_xlabel(r'$\boldsymbol{S}$')
    ax2.set_ylabel(r'$\boldsymbol{N_{\lambda}}$')
    ax2.legend(loc='best')
    fig.tight_layout(rect=[.02,.02,1,.98])
    for ext in ('.pdf','.png'):
        fig.savefig(str(output)+ext,dpi=DPI,bbox_inches='tight',pad_inches=.02)
        print('Saved',str(output)+ext)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode',choices=('all','rotate','unrotate','plot'),default='all')
    parser.add_argument('--input',choices=('structured','saved'),default='structured')
    parser.add_argument('--dimension',type=int,default=D_DEFAULT)
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('.'))
    parser.add_argument('--shots',type=int,default=None)
    parser.add_argument('--trials',type=int,default=None)
    parser.add_argument('--eta',type=float,default=ETA)
    parser.add_argument('--kappa',type=float,default=KAPPA)
    parser.add_argument('--eps',type=float,default=EPS)
    parser.add_argument('--S',type=int,nargs='+',default=list(S_LIST))
    parser.add_argument('--projector',choices=('rank_one','first_S'),default='rank_one')
    parser.add_argument('--first-half',action='store_true')
    parser.add_argument('--mean-tail',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.shots is None or args.trials is None:
        param_file=args.data_dir/'para_list.npy'
        if param_file.exists():
            para=np.load(param_file)
            if args.shots is None: args.shots=int(para[1])
            if args.trials is None: args.trials=int(para[2])
    if args.shots is None: args.shots=10000
    if args.trials is None: args.trials=10
    if args.shots<1 or args.trials<1:
        raise ValueError('Shots and trials must be positive.')
    rotated=args.output_dir/'prdm_rotate.csv'
    unrotated=args.output_dir/'prdm_unrotate.csv'
    if args.mode!='plot':
        lam,V0=load_state(args.data_dir,args.input,args.dimension)
        d,r=V0.shape
        Ss=sorted(set(int(x) for x in args.S))
        for S in Ss:
            if S<r or S>d:
                raise ValueError(f'S={S} must satisfy chi<=S<=d.')
            projector_indices(S,d,r,args.projector)
            if S!=2 and not (S%2 and is_prime(S)):
                raise ValueError(f'S={S} is not supported; select an odd prime or 2.')
        print(f'Input: {args.input}, d={d}, chi={r}, shots={args.shots}, trials={args.trials}')
        if args.mode in ('all','rotate'):
            simulate_rotated(lam,V0,Ss,args.shots,args.trials,args.eta,
                             args.kappa,args.eps,args.projector,rotated)
        if args.mode in ('all','unrotate'):
            simulate_unrotated(lam,V0,Ss,args.shots,args.trials,args.eta,
                               args.kappa,args.projector,unrotated)
        chi=r
    else:
        if args.input=='saved':
            lam,_=load_state(args.data_dir,args.input,args.dimension)
            chi=len(lam)
        else: chi=4
    if args.mode in ('all','plot'):
        draw_plot(rotated,unrotated,chi,args.output_dir/'PRDM',
                  first_half=args.first_half,use_max_tail=not args.mean_tail)


if __name__=='__main__':
    main()
