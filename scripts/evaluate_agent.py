"""
evaluate_agent.py
=================
Evaluates the trained RL agent and all baselines on any data source,
computing all performance metrics.

New metrics added
-----------------
  mean_selected        — average number of metrics selected per window
                         absolute count, e.g. 15.4 out of N*M=150
  selection_rate       — mean_selected / (N*M), e.g. 0.103 = 10.3%
                         shows how selective the agent is vs baselines
  mean_collection_cost — mean tier-weighted monitoring cost per window
                         = Σ_n cost_n × selected_n
                         reflects IoT/Edge/Cloud cost structure

These three join the PRIMARY metrics table and are saved to CSV.

Coverage mode (--coverage-mode)
--------------------------------
  window       — correct, matches MonitoringEnv. Use for paper results.
  per_timestep — legacy, inflated. Use to compare with old training logs.
"""

import os
import sys
import argparse
import csv
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_data import build_episode_pool, load_dataset, save_dataset, topology_kwargs, assign_tiers, pool_dir, pool_config_key
from utils.seeding import resolve_seeds
from envs.monitoring_env import MonitoringEnv
from agents.policy import ActorCritic, PerMetricActorCritic
from baselines.baselines import MonitorEverything, RandomFeasible, TopKAnomalyCausal


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",        default="pkl",
                   choices=["synthetic","pkl","npy","smd"])
    p.add_argument("--config",        default="configs/config.yaml")
    p.add_argument("--run-name",      default=None)
    p.add_argument("--model",         default=None)
    p.add_argument("--out-dir",       default=None)
    p.add_argument("--episodes",      type=int, default=None)
    p.add_argument("--save-detail",   action="store_true")
    p.add_argument("--no-agent",      action="store_true")
    p.add_argument("--no-baselines",  action="store_true")
    p.add_argument("--device",        default=None)
    p.add_argument("--coverage-mode", default=None,
                   choices=["window","per_timestep"],
                   help="'window' (correct, for paper) or "
                        "'per_timestep' (legacy, inflated)")
    # pkl
    p.add_argument("--pkl-path",       default=None)
    # npy
    p.add_argument("--anomaly-scores", default=None)
    p.add_argument("--causal-scores",  default=None)
    p.add_argument("--labels",         default=None)
    p.add_argument("--dataset-name",   default="custom")
    # smd
    p.add_argument("--anomaly-dir",   default="data/anomaly_scores")
    p.add_argument("--causal-dir",    default="data/causal_scores")
    p.add_argument("--machines",      nargs="+", default=None)
    p.add_argument("--n-metrics",     type=int, default=38)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════



# assign_tiers is imported from data.synthetic_data — it reads the authoritative
# node_tier_labels recorded at generation time, and only falls back to cost
# thresholds for legacy pools.  Do not reintroduce a local threshold version:
# it silently misattributes every per-tier metric once costs are varied.


# ------------------------------------------------------------------
# Policy factory
# ------------------------------------------------------------------

def build_policy(cfg: dict, obs_dim: int, action_dim: int) -> torch.nn.Module:
    """
    Instantiate the correct policy class based on config.

    Reads cfg["policy"]["policy_type"]:
      "flat"       (default) — ActorCritic flat MLP
      "per_metric"           — PerMetricActorCritic shared encoder

    Falls back to "flat" if policy_type is not set, so existing configs
    that do not specify policy_type continue to work unchanged.
    """
    pcfg        = cfg["policy"]
    policy_type = pcfg.get("policy_type", "flat")
    ecfg        = cfg["environment"]
    N, M        = ecfg["n_nodes"], ecfg["n_metrics"]
    feature_dim = obs_dim // (N * M)   # = 6 with current monitoring_env

    if policy_type == "per_metric":
        policy = PerMetricActorCritic(
            n_nodes        = N,
            n_metrics      = M,
            feature_dim    = feature_dim,
            metric_hidden  = pcfg.get("metric_hidden",  128),
            global_hidden  = pcfg.get("global_hidden",  256),
            use_layer_norm = pcfg.get("use_layer_norm", True),
            global_pool    = pcfg.get("global_pool",    "mean"),
        )
        print(f"[policy] PerMetricActorCritic  "
              f"mh={pcfg.get('metric_hidden',128)}  "
              f"gh={pcfg.get('global_hidden',256)}  "
              f"pool={pcfg.get('global_pool','mean')}")
    else:
        policy = ActorCritic(
            obs_dim        = obs_dim,
            action_dim     = action_dim,
            hidden_sizes   = pcfg.get("hidden_sizes", [512, 256]),
            use_layer_norm = pcfg.get("use_layer_norm", True),
        )
        print(f"[policy] ActorCritic  "
              f"hidden={pcfg.get('hidden_sizes',[512,256])}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[policy] Parameters: {n_params:,}")
    return policy


# ══════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════

def load_synthetic(cfg, n_episodes, episode_len):
    ecfg = cfg["environment"]; dcfg = cfg["data"]
    eps  = build_episode_pool(
        n_episodes=n_episodes,
        n_nodes=ecfg["n_nodes"], n_metrics=ecfg["n_metrics"],
        episode_length=episode_len or ecfg["episode_length"],
        anomaly_density=dcfg["anomaly_density"]*1.2,
        causal_density=dcfg["causal_density"],
        base_seed=dcfg["seed"]+99999,
        anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
        causal_baseline_pct=ecfg["causal_baseline_pct"],
        **topology_kwargs(ecfg),
    )
    for ep in eps: ep["name"] = "synthetic"
    return eps


def load_pkl(pkl_path):
    with open(pkl_path,"rb") as f: eps = pickle.load(f)
    for ep in eps: ep.setdefault("name", Path(pkl_path).stem)
    print(f"[loader] {len(eps)} episodes ← {pkl_path}")
    return eps


def load_npy(anomaly_path, causal_path, labels_path, name, cfg):
    A = np.load(anomaly_path).astype(np.float32)
    C = np.load(causal_path).astype(np.float32)
    L = np.load(labels_path).astype(np.int32) if labels_path else None
    if A.ndim==2: A=A[:,np.newaxis,:]
    if C.ndim==1: C=C[np.newaxis,:]
    if L is not None and L.ndim==2: L=L[:,np.newaxis,:]
    T,N,M = A.shape
    if L is None: L=np.zeros((T,N,M),dtype=np.int32)
    ecfg = cfg["environment"]
    caps = np.array([max(2,int(M*0.5))]*N, dtype=np.float32)
    return [{"anomaly_scores":A,"anomaly_labels":L,"causal_scores":C,
             "node_capacities":caps,
             "node_costs":np.full(N,ecfg["per_metric_cost"],dtype=np.float32),
             "n_nodes":N,"n_metrics":M,"T":T,"name":name,
             "b_a":float(np.percentile(A,ecfg["anomaly_baseline_pct"])),
             "b_c":float(np.percentile(C,ecfg["causal_baseline_pct"]))}]


def load_smd(anomaly_dir, causal_dir, machines, n_metrics, cfg):
    anomaly_dir=Path(anomaly_dir); causal_dir=Path(causal_dir)
    ecfg=cfg["environment"]
    if machines is None:
        machines=sorted([f.stem.replace("_scores","")
                         for f in anomaly_dir.glob("*_scores.npy")])
    episodes=[]
    for machine in machines:
        sp=anomaly_dir/f"{machine}_scores.npy"
        lp=anomaly_dir/f"{machine}_labels.npy"
        cp=causal_dir/f"{machine}_causal.npy"
        if not sp.exists(): continue
        A=np.load(sp).astype(np.float32); T,M=A.shape
        L=np.load(lp).astype(np.int32)
        C=(np.load(cp).astype(np.float32) if cp.exists()
           else np.full(n_metrics,0.5,dtype=np.float32))
        grp=int(machine.split("-")[1])
        cap=float(int(M*(0.7 if grp==1 else 0.5 if grp==2 else 0.3)))
        episodes.append({
            "anomaly_scores":A[:,np.newaxis,:],"anomaly_labels":np.stack([L]*M,axis=1)[:,np.newaxis,:],
            "causal_scores":C[np.newaxis,:],"node_capacities":np.array([cap],dtype=np.float32),
            "node_costs":np.array([ecfg["per_metric_cost"]],dtype=np.float32),
            "n_nodes":1,"n_metrics":M,"T":T,"name":machine,
            "b_a":float(np.percentile(A,ecfg["anomaly_baseline_pct"])),
            "b_c":float(np.percentile(C,ecfg["causal_baseline_pct"])),
        })
        print(f"[loader] {machine}: T={T} M={M} cap={cap:.0f}")
    return episodes


# ══════════════════════════════════════════════════════════════════
# Trajectory collectors
# ══════════════════════════════════════════════════════════════════

def collect_agent_trajectory(policy, episode, reward_weights, cfg, device):
    ecfg = cfg["environment"]
    env  = MonitoringEnv(
        episode_pool=[episode], reward_weights=reward_weights,
        per_metric_cost=ecfg["per_metric_cost"],
        anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
        causal_baseline_pct=ecfg["causal_baseline_pct"],
        randomize_episodes=False,
        window_size=ecfg.get("window_size",1),
        window_agg=ecfg.get("window_agg","max"),
        coverage_mode=ecfg.get("coverage_mode","window"),
    )
    obs, _ = env.reset()
    done   = False
    actions = []
    N=episode["n_nodes"]; M=episode["n_metrics"]
    policy.eval()
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
            action = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
            obs,_,term,trunc,_ = env.step(action)
            done = term or trunc
            actions.append(action.reshape(N,M).copy())
    return actions


def collect_baseline_trajectory(baseline, episode, window_size, window_agg="max"):
    N=episode["n_nodes"]; M=episode["n_metrics"]
    T=episode["T"]; W=window_size
    A=episode["anomaly_scores"]; C=episode["causal_scores"]
    n_windows=T//W

    agg=np.zeros((n_windows,N,M),dtype=np.float32)
    for w in range(n_windows):
        t_s,t_e=w*W,min((w+1)*W,T)
        agg[w]=(A[t_s:t_e].max(axis=0) if window_agg=="max"
                else A[t_s:t_e].mean(axis=0))

    proxy={"n_nodes":N,"n_metrics":M,"T":n_windows,"anomaly_scores":agg,
           "anomaly_labels":np.zeros((n_windows,N,M),dtype=np.int32),
           "causal_scores":C,"node_capacities":episode["node_capacities"],
           "b_a":episode.get("b_a",0.0),"b_c":episode.get("b_c",0.0)}
    baseline.reset(proxy)
    return [baseline.select(w).copy() for w in range(n_windows)]


# ══════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════

def compute_episode_metrics(actions, episode, window_size,
                             coverage_mode="window"):
    """
    Compute all performance metrics for one episode.

    Parameters
    ----------
    actions       : list of [N,M] binary arrays, one per window
    episode       : episode dict
    window_size   : W
    coverage_mode : "window" (correct) or "per_timestep" (legacy)
    """
    A     = episode["anomaly_scores"]
    L     = episode["anomaly_labels"]
    C     = episode["causal_scores"]
    caps  = episode["node_capacities"]
    costs = episode.get("node_costs",
                        np.full(episode["n_nodes"],0.01,dtype=np.float32))
    N,M   = episode["n_nodes"],episode["n_metrics"]
    T,W   = episode["T"],window_size
    n_windows    = len(actions)
    tiers        = assign_tiers(episode)   # authoritative labels, not cost thresholds
    total_budget = float(caps.sum())
    total_metrics = N * M
    b_a           = episode.get("b_a",0.0)
    sel_freq      = np.stack(actions).mean(axis=0)

    # ── Coverage helper ─────────────────────────────────────────
    def _window_cov(w, a):
        t_s, t_e = w*W, min((w+1)*W, T)
        if coverage_mode == "window":
            L_w = L[t_s:t_e].max(axis=0)
            n_an = int(L_w.sum())
            return (float((a*L_w).sum()/n_an) if n_an>0 else 1.0), L_w
        else:
            covs = []
            for t in range(t_s, t_e):
                L_t=L[t]; n_an=int(L_t.sum())
                covs.append(float((a*L_t).sum()/n_an) if n_an>0 else 1.0)
            L_w = L[t_s:t_e].max(axis=0)
            return float(np.mean(covs)), L_w

    # ── Group 1: Coverage ───────────────────────────────────────
    acr_vals=[]; wacr_n=wacr_d=0.0; fmr_vals=[]
    for w,a in enumerate(actions):
        t_s,t_e=w*W,min((w+1)*W,T)
        cov,L_w=_window_cov(w,a)
        A_w=A[t_s:t_e].max(axis=0)
        acr_vals.append(cov)
        wacr_n+=float((a*L_w*A_w).sum()); wacr_d+=float((L_w*A_w).sum())
        sel=float(a.sum())
        if sel>0: fmr_vals.append(float((a*(1-L_w)).sum())/sel)
    acr=float(np.mean(acr_vals))
    weighted_acr=wacr_n/wacr_d if wacr_d>1e-8 else 1.0
    fmr=float(np.mean(fmr_vals)) if fmr_vals else 0.0
    prec=1.0-fmr; f1=2*acr*prec/(acr+prec+1e-8)

    # ── Group 2: Detection speed ────────────────────────────────
    latencies=[]; ttfd_events=[]
    for n in range(N):
        for m in range(M):
            lbl=L[:,n,m]
            if lbl.sum()==0: continue
            in_ev=False; t_st=0
            for t in range(T):
                if lbl[t]==1 and not in_ev: in_ev=True; t_st=t
                elif lbl[t]==0 and in_ev:
                    in_ev=False; esw=t_st//W
                    dw=next((w for w in range(esw,n_windows) if actions[w][n,m]==1),None)
                    lat=(dw-esw) if dw is not None else n_windows
                    latencies.append(lat); ttfd_events.append(lat<=1)
            if in_ev:
                esw=t_st//W
                dw=next((w for w in range(esw,n_windows) if actions[w][n,m]==1),None)
                lat=(dw-esw) if dw is not None else n_windows
                latencies.append(lat); ttfd_events.append(lat<=1)
    latency_mean  =float(np.mean(latencies))          if latencies else 0.0
    latency_median=float(np.median(latencies))        if latencies else 0.0
    latency_p90   =float(np.percentile(latencies,90)) if latencies else 0.0
    ttfd          =float(np.mean(ttfd_events))        if ttfd_events else 1.0

    # ── Group 3: Selection & cost (NEW) ────────────────────────
    sel_counts=[]    # absolute selected per window
    cost_per_w=[]    # tier-weighted cost per window
    bur_v=[]; cvr_v=[]

    for a in actions:
        spn=a.sum(axis=1)                                    # [N] selected per node
        n_sel=int(a.sum())
        sel_counts.append(n_sel)
        cost_per_w.append(float((costs*spn).sum()))          # Σ_n cost_n*sel_n
        bur_v.append(float(n_sel)/(total_budget+1e-8))
        cvr_v.append(float((spn>caps).mean()))

    mean_selected        = float(np.mean(sel_counts))
    selection_rate       = mean_selected / (total_metrics + 1e-8)
    mean_collection_cost = float(np.mean(cost_per_w))
    bur  = float(np.mean(bur_v))
    cvr  = float(np.mean(cvr_v))
    cuc  = acr / (mean_collection_cost + 1e-8)

    # ── Group 4: Selection quality ──────────────────────────────
    tp_v=[]; sg_v=[]
    for w,a in enumerate(actions):
        t_s,t_e=w*W,min((w+1)*W,T)
        A_w=A[t_s:t_e].max(axis=0)
        for n in range(N):
            k=int(a[n].sum())
            if k==0: continue
            tp_v.append(len(set(np.argsort(A_w[n])[::-1][:k]) &
                            set(np.where(a[n]==1)[0]))/k)
        ns=a.sum(); nu=total_metrics-ns
        if ns>0 and nu>0:
            sg_v.append(float(A_w[a==1].mean())-float(A_w[a==0].mean()))
    topk_precision=float(np.mean(tp_v)) if tp_v else 0.0
    score_gap=float(np.mean(sg_v)) if sg_v else 0.0
    try:
        from scipy.stats import spearmanr
        rhos=[spearmanr(sel_freq[n],C[n])[0]
              for n in range(N) if sel_freq[n].std()>1e-6 and C[n].std()>1e-6]
        causal_align=float(np.mean(rhos)) if rhos else float("nan")
    except ImportError:
        causal_align=float("nan")

    # ── Group 5: Stability ──────────────────────────────────────
    stab_v=[]; react_v=[]
    for w in range(1,n_windows):
        stab_v.append(1.0-np.abs(actions[w]-actions[w-1]).sum()/(total_metrics))
    stability=float(np.mean(stab_v)) if stab_v else 1.0
    for n in range(N):
        for m in range(M):
            for w in range(1,n_windows-1):
                t_s=w*W
                if A[t_s:min(t_s+W,T),n,m].max()>b_a:
                    react_v.append(float(actions[w][n,m])-float(actions[w-1][n,m]))
    reactivity=float(np.mean(react_v)) if react_v else 0.0

    # ── Group 6: Tier-specific ──────────────────────────────────
    tier_m={}
    for tier in ("iot","edge","cloud"):
        idx=[i for i,t in enumerate(tiers) if t==tier]
        if not idx:
            for k in (f"acr_{tier}",f"cvr_{tier}",f"cuc_{tier}",
                      f"selected_{tier}",f"cost_{tier}"):
                tier_m[k]=float("nan")
            continue
        ta=[]; tv=[]; tc=[]; tsel=[]
        for w,a in enumerate(actions):
            t_s,t_e=w*W,min((w+1)*W,T)
            for n in idx:
                if coverage_mode=="window":
                    L_wn=L[t_s:t_e,n,:].max(axis=0); n_an=int(L_wn.sum())
                    ta.append(float((a[n]*L_wn).sum()/n_an) if n_an>0 else 1.0)
                else:
                    node_covs=[]
                    for t in range(t_s,t_e):
                        L_tn=L[t,n,:]; n_an=int(L_tn.sum())
                        node_covs.append(float((a[n]*L_tn).sum()/n_an) if n_an>0 else 1.0)
                    ta.append(float(np.mean(node_covs)))
                tv.append(float(a[n].sum()>caps[n]))
                tc.append(float(a[n].sum()*costs[n]))
                tsel.append(float(a[n].sum()))
        t_acr=float(np.mean(ta)); t_cost=float(np.mean(tc))
        tier_m[f"acr_{tier}"]      = t_acr
        tier_m[f"cvr_{tier}"]      = float(np.mean(tv))
        tier_m[f"cuc_{tier}"]      = t_acr/(t_cost+1e-8)
        tier_m[f"selected_{tier}"] = float(np.mean(tsel))   # avg per node
        tier_m[f"cost_{tier}"]     = t_cost

    return {
        # Coverage
        "acr":acr,"weighted_acr":weighted_acr,"fmr":fmr,"f1_coverage":f1,
        # Detection
        "latency_mean":latency_mean,"latency_median":latency_median,
        "latency_p90":latency_p90,"ttfd":ttfd,
        # Selection & cost (NEW)
        "mean_selected":mean_selected,
        "selection_rate":selection_rate,
        "mean_collection_cost":mean_collection_cost,
        # Resource efficiency
        "bur":bur,"cuc":cuc,"cvr":cvr,
        # Quality
        "topk_precision":topk_precision,"score_gap":score_gap,
        "causal_align":causal_align,
        # Stability
        "stability":stability,"reactivity":reactivity,
        "coverage_mode":coverage_mode,
        **tier_m,
    }


# ══════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════

def aggregate(ep_metrics):
    skip={"coverage_mode"}
    agg={}
    for k in ep_metrics[0]:
        if k in skip: continue
        vals=[m[k] for m in ep_metrics
              if not (isinstance(m[k],float) and np.isnan(m[k]))]
        agg[f"{k}_mean"]=float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] =float(np.std(vals))  if vals else float("nan")
    return agg


# ══════════════════════════════════════════════════════════════════
# Print tables
# ══════════════════════════════════════════════════════════════════

PRIMARY = [
    ("acr_mean",                 "ACR ↑",         ".4f"),
    ("weighted_acr_mean",        "wACR ↑",        ".4f"),
    ("fmr_mean",                 "FMR ↓",         ".4f"),
    ("f1_coverage_mean",         "F1-Cov ↑",      ".4f"),
    ("latency_mean_mean",        "Latency ↓",     ".2f"),
    ("ttfd_mean",                "TTFD@1 ↑",      ".4f"),
    # Selection & cost
    ("mean_selected_mean",       "Selected ↓",    ".1f"),
    ("selection_rate_mean",      "Sel.Rate ↓",    ".3f"),
    ("mean_collection_cost_mean","CollCost ↓",    ".5f"),
    # Efficiency
    ("bur_mean",                 "BUR",            ".4f"),
    ("cuc_mean",                 "CUC ↑",          ".4f"),
    ("cvr_mean",                 "CVR ↓",          ".4f"),
]

SECONDARY = [
    ("topk_precision_mean",  "TopK-P ↑",      ".4f"),
    ("score_gap_mean",       "ScoreGap ↑",    ".4f"),
    ("causal_align_mean",    "CausalAlign ↑", ".4f"),
    ("stability_mean",       "Stability ↑",   ".4f"),
    ("reactivity_mean",      "Reactivity ↑",  ".4f"),
]

TIER = [
    ("acr_iot_mean",       "ACR-IoT ↑",        ".4f"),
    ("acr_edge_mean",      "ACR-Edge ↑",       ".4f"),
    ("acr_cloud_mean",     "ACR-Cloud ↑",      ".4f"),
    ("cvr_iot_mean",       "CVR-IoT ↓",        ".4f"),
    ("cvr_edge_mean",      "CVR-Edge ↓",       ".4f"),
    ("cvr_cloud_mean",     "CVR-Cloud ↓",      ".4f"),
    ("selected_iot_mean",  "Sel-IoT ↓",        ".2f"),
    ("selected_edge_mean", "Sel-Edge ↓",       ".2f"),
    ("selected_cloud_mean","Sel-Cloud ↓",      ".2f"),
    ("cost_iot_mean",      "Cost-IoT ↓",       ".5f"),
    ("cost_edge_mean",     "Cost-Edge ↓",      ".5f"),
    ("cost_cloud_mean",    "Cost-Cloud ↓",     ".5f"),
]


def _v(d, k, f):
    v = d.get(k, float("nan"))
    if isinstance(v, float) and np.isnan(v): return "  —  "
    # Show selection_rate as percentage
    if "selection_rate" in k: return f"{v*100:.1f}%"
    return format(v, f)


def print_table(title, results, defs):
    methods=list(results.keys())
    cw=max(15,max(len(m) for m in methods)+2)
    w=22+cw*len(methods)
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")
    print(f"{'Metric':<22}"+"".join(f"{m:>{cw}}" for m in methods))
    print("─"*w)
    for k,label,fmt in defs:
        print(f"{label:<22}"+"".join(f"{_v(results[m],k,fmt):>{cw}}"
                                      for m in methods))
    print("═"*w)


# ══════════════════════════════════════════════════════════════════
# CSV / plot saving
# ══════════════════════════════════════════════════════════════════

def save_aggregate_csv(results, path):
    keys=sorted({k for v in results.values() for k in v})
    rows=[{"method":n,**{k:v.get(k,float("nan")) for k in keys}}
          for n,v in results.items()]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method"]+keys)
        w.writeheader(); w.writerows(rows)
    print(f"[eval] Aggregate → {path}")


def save_detail_csv(detail, path):
    keys=sorted({k for rows in detail.values() for r in rows for k in r})
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","episode"]+keys)
        w.writeheader()
        for method,rows in detail.items():
            for i,r in enumerate(rows):
                w.writerow({"method":method,"episode":i,
                            **{k:r.get(k,float("nan")) for k in keys}})
    print(f"[eval] Detail    → {path}")


def save_latency_csv(detail, path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["method","episode","latency_mean","latency_median",
                    "latency_p90","ttfd"])
        for method,rows in detail.items():
            for i,r in enumerate(rows):
                w.writerow([method,i,r["latency_mean"],r["latency_median"],
                            r["latency_p90"],r["ttfd"]])
    print(f"[eval] Latency   → {path}")


def plot_results(results, out_dir):
    try: import matplotlib.pyplot as plt
    except ImportError: return
    methods=list(results.keys())
    metrics=["acr_mean","f1_coverage_mean","mean_selected_mean",
             "mean_collection_cost_mean"]
    labels =["ACR","F1-Coverage","Mean Selected","Collection Cost"]
    colors =["#2196F3","#4CAF50","#FF5722","#9C27B0","#FF9800"]
    fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),4))
    for ax,metric,label in zip(axes,metrics,labels):
        vals=[results[m].get(metric,0) for m in methods]
        ax.bar(range(len(methods)),vals,
               color=[colors[i%len(colors)] for i in range(len(methods))],alpha=0.85)
        ax.set_title(label)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods,rotation=20,ha="right",fontsize=8)
        ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout()
    out=os.path.join(out_dir,"evaluation_comparison.png")
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close()
    print(f"[eval] Plot      → {out}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args=parse_args()
    import yaml
    with open(args.config) as f: cfg=yaml.safe_load(f)

    device=torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu"))

    ecfg=cfg["environment"]; dcfg=cfg["data"]; rw=cfg["reward"]
    N,M=ecfg["n_nodes"],ecfg["n_metrics"]
    W=ecfg.get("window_size",1); agg=ecfg.get("window_agg","max")

    coverage_mode=(args.coverage_mode or ecfg.get("coverage_mode","window"))
    ecfg["coverage_mode"]=coverage_mode

    reward_weights={"alpha":rw["alpha"],"beta":rw["beta"],
                    "lambda_":rw["lambda_"],"delta":rw["delta"],"eta":rw["eta"]}

    base_results=cfg["paths"]["results_dir"]
    run_dir=(os.path.join(base_results,args.run_name)
             if args.run_name else base_results)
    out_dir=args.out_dir or os.path.join(run_dir,"metrics")
    os.makedirs(out_dir,exist_ok=True)

    print(f"\n[eval] Source        : {args.source}")
    print(f"[eval] Coverage mode : {coverage_mode}")
    print(f"[eval] N={N}  M={M}  N*M={N*M}  W={W}")

    if args.source=="synthetic":
        episodes=load_synthetic(cfg,args.episodes or 20,None)
    elif args.source=="pkl":
        path=args.pkl_path or os.path.join(
            pool_dir(cfg["paths"]["data_dir"], N, M, resolve_seeds(cfg)[0],
                          config_key=pool_config_key(cfg)),"eval_episodes.pkl")
        episodes=load_pkl(path)
    elif args.source=="npy":
        episodes=load_npy(args.anomaly_scores,args.causal_scores,
                          args.labels,args.dataset_name,cfg)
    elif args.source=="smd":
        episodes=load_smd(args.anomaly_dir,args.causal_dir,
                          args.machines,args.n_metrics,cfg)

    n_eval=args.episodes or len(episodes)
    episodes=episodes[:n_eval]
    print(f"[eval] Episodes      : {len(episodes)}")
    print(f"[eval] Output        : {out_dir}")

    methods: Dict[str,callable]={}

    if not args.no_baselines:
        for name,bl in [("Monitor-Everything",MonitorEverything()),
                         ("Random-Feasible",RandomFeasible(seed=resolve_seeds(cfg)[1])),
                         ("TopK-Anomaly-Causal",TopKAnomalyCausal())]:
            methods[name]=(lambda ep,bl=bl:
                           collect_baseline_trajectory(bl,ep,W,agg))

    if not args.no_agent:
        model_path=(args.model or
                    (os.path.join(run_dir,cfg["paths"]["model_name"])
                     if args.run_name else None))
        if model_path and os.path.exists(model_path):
            _tmp=MonitoringEnv(
                episode_pool=episodes[:1],reward_weights=reward_weights,
                per_metric_cost=ecfg["per_metric_cost"],
                anomaly_baseline_pct=ecfg["anomaly_baseline_pct"],
                causal_baseline_pct=ecfg["causal_baseline_pct"],
                window_size=W,window_agg=agg,coverage_mode=coverage_mode)
            obs_dim=_tmp.observation_space.shape[0]
            action_dim=_tmp.action_space.n
            policy=build_policy(cfg, obs_dim, action_dim)
            policy.load_state_dict(torch.load(model_path,map_location=device))
            policy.to(device)
            print(f"[eval] Model         : {model_path}")
            print(f"[eval] obs_dim={obs_dim}  action_dim={action_dim}")
            methods["RL-PPO (Ours)"]=lambda ep: collect_agent_trajectory(
                policy,ep,reward_weights,cfg,device)
        else:
            print("[eval] No model found — baselines only")

    if not methods:
        print("[eval] Nothing to run."); return

    agg_results={}; detail={}
    for method_name,collect_fn in methods.items():
        print(f"\n[eval] {method_name} ...")
        ep_metrics=[]
        for i,ep in enumerate(episodes):
            actions=collect_fn(ep)
            m=compute_episode_metrics(actions,ep,W,coverage_mode)
            ep_metrics.append(m)
            if (i+1)%max(1,n_eval//5)==0 or i==n_eval-1:
                print(f"       ep {i+1}/{n_eval}  "
                      f"ACR={m['acr']:.3f}  F1={m['f1_coverage']:.3f}  "
                      f"Sel={m['mean_selected']:.1f}({m['selection_rate']*100:.1f}%)  "
                      f"Cost={m['mean_collection_cost']:.4f}  "
                      f"Lat={m['latency_mean']:.1f}w  [{coverage_mode}]")
        agg_results[method_name]=aggregate(ep_metrics)
        detail[method_name]=ep_metrics

    suffix=f"(coverage_mode={coverage_mode})"
    print_table(f"Primary Metrics {suffix}",             agg_results,PRIMARY)
    print_table(f"Selection Quality & Stability {suffix}",agg_results,SECONDARY)
    print_table(f"Tier-Specific Analysis {suffix}",       agg_results,TIER)

    stem=f"metrics_{coverage_mode}"
    save_aggregate_csv(agg_results,os.path.join(out_dir,f"{stem}_aggregate.csv"))
    save_latency_csv(detail,        os.path.join(out_dir,f"{stem}_latency.csv"))
    if args.save_detail:
        save_detail_csv(detail,     os.path.join(out_dir,f"{stem}_detail.csv"))
    plot_results(agg_results,out_dir)
    print(f"\n[eval] Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()