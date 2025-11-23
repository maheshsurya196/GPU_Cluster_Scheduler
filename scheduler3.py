# ============================================================
#  CHUNK 1 — Imports, Constants, Paged Queue, Process State
# ============================================================

import torch
import torch.nn as nn
from typing import Dict
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAGE_SIZE = 256


# ============================================================
#  PagedQueue — Deterministic GPU Queue with Paged Allocation
# ============================================================

class PagedQueue:
    """
    Deterministic VRAM queue using fixed-size pages.
    No Python lists inside GPU loops.
    """

    def __init__(self, entry_dim: int):
        self.entry_dim = entry_dim
        self.pages = []          # list[Tensor(PAGE_SIZE, entry_dim)]
        self.free_pages = []

        self.head = 0
        self.tail = 0

    def _ensure_page(self, page_idx: int):
        while len(self.pages) <= page_idx:
            if self.free_pages:
                self.pages.append(self.free_pages.pop())
            else:
                self.pages.append(torch.zeros((PAGE_SIZE, self.entry_dim),
                                              device=DEVICE))

    def push(self, t: torch.Tensor):
        page_idx = self.tail // PAGE_SIZE
        offset = self.tail % PAGE_SIZE
        self._ensure_page(page_idx)
        self.pages[page_idx][offset].copy_(t)
        self.tail += 1

    def pop(self):
        if self.empty():
            return None

        page_idx = self.head // PAGE_SIZE
        offset   = self.head % PAGE_SIZE

        out = self.pages[page_idx][offset].clone()
        self.head += 1

        if offset == PAGE_SIZE - 1:
            # reclaim whole page
            self.free_pages.append(self.pages[page_idx])
            self.pages[page_idx] = None

        return out

    def empty(self):
        return self.head >= self.tail

    def size(self):
        return self.tail - self.head


# ============================================================
#  ProcessState — All scheduler state in VRAM
# ============================================================

class ProcessState:
    """
    Holds all relevant GPU tensors for the scheduler.
    No learned weights. Pure deterministic state.
    """

    def __init__(self, batch: int, num_procs: int,
                 state_dim: int, num_cores: int):

        self.process_states = torch.randn(
            batch, num_procs, state_dim, device=DEVICE)

        self.core_states = torch.randn(
            batch, num_cores, 16, device=DEVICE)

        self.system_load = torch.randn(
            batch, 32, device=DEVICE)
# ============================================================
#  CHUNK 2 — Deterministic Routing + Parallel Policy Scoring
# ============================================================


class DeterministicPolicyRouter(nn.Module):
    """
    Deterministic routing:
      - No learning, no randomness.
      - Program-as-weights style: fixed routing matrix + bias.
      - Selects exactly ONE policy per (batch, process) index.

    Conceptually:
        routing_scores = process_feats @ W^T + b
        policy_id = argmax(routing_scores, dim=-1)
    """

    def __init__(self, state_dim: int, num_policies: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_policies = num_policies

        # Fixed routing weights (as if compiler wrote them).
        # Shape: [P, D]
        base = torch.linspace(0.5, 1.5, steps=num_policies).view(num_policies, 1)
        col_scale = torch.linspace(1.0, 0.1, steps=state_dim).view(1, state_dim)
        routing_matrix = base * col_scale  # simple rank-1 deterministic pattern

        # Fixed bias per policy
        bias = torch.linspace(-0.2, 0.2, steps=num_policies)

        self.register_buffer("routing_matrix", routing_matrix)  # [P, D]
        self.register_buffer("bias", bias)                      # [P]

    def forward(self, process_feats: torch.Tensor):
        """
        process_feats: [B, N, D]

        Returns:
          selected_policies: [B, N] (long)
          policy_mask:      [B, N, P] (one-hot)
        """
        B, N, D = process_feats.shape
        P = self.num_policies

        # [B, N, D] x [D, P] -> [B, N, P]
        logits = torch.matmul(process_feats, self.routing_matrix.t())  # [B, N, P]
        logits = logits + self.bias.view(1, 1, P)

        selected = torch.argmax(logits, dim=-1)  # [B, N]

        # one-hot mask: [B, N, P]
        policy_mask = torch.zeros(B, N, P, device=process_feats.device)
        policy_mask.scatter_(-1, selected.unsqueeze(-1), 1.0)

        return selected, policy_mask


class ParallelPolicyScorer(nn.Module):
    """
    Deterministic, fully parallel policy scoring.

    No parameters, no learning.
    Each policy is a different fixed functional of the process features:

      Policy 0: mean of all dims
      Policy 1: mean of first half
      Policy 2: mean of second half
      Policy 3: variance across dims
      Policy 4: negative mean absolute value

    All vectorized over batch/process.
    """

    def __init__(self, state_dim: int, num_policies: int):
        super().__init__()
        self.state_dim = state_dim
        self.num_policies = num_policies

    def forward(self, process_feats: torch.Tensor) -> torch.Tensor:
        """
        process_feats: [B, N, D]

        Returns:
          policy_scores: [B, N, P]
        """
        B, N, D = process_feats.shape
        P = self.num_policies
        assert P >= 5, "This simple scorer assumes at least 5 policies."

        # Basic statistics
        mean_all = process_feats.mean(dim=-1)                 # [B, N]
        mean_first = process_feats[..., : D // 2].mean(-1)    # [B, N]
        mean_second = process_feats[..., D // 2 :].mean(-1)   # [B, N]
        var_all = process_feats.var(dim=-1, unbiased=False)   # [B, N]
        neg_abs_mean = -process_feats.abs().mean(dim=-1)      # [B, N]

        base_list = [mean_all, mean_first, mean_second, var_all, neg_abs_mean]

        # If num_policies > 5, derive extra deterministic variants
        scores = []
        for i in range(P):
            if i < 5:
                scores.append(base_list[i])
            else:
                # simple deterministic reweighting
                s = mean_all * (1.0 + 0.05 * i) - 0.1 * var_all
                scores.append(s)

        policy_scores = torch.stack(scores, dim=-1)  # [B, N, P]
        return policy_scores
# ============================================================
#  CHUNK 3 — Deterministic Loop-Unrolled Refiners
# ============================================================

class MLFQRefiner(nn.Module):
    """
    Deterministic unrolled refinement for MLFQ.

    Each iteration corresponds to one "loop step" of:
        for iter in range(K):
            adjust priority / fairness / burst-age interaction

    No weights, no attention, no randomness.
    """
    def __init__(self, state_dim: int, steps: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.steps = steps

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, N, D]
        """
        x = feats
        B, N, D = x.shape

        for i in range(self.steps):
            # local stats per process
            mean_local = x.mean(dim=-1, keepdim=True)      # [B,N,1]
            var_local = (x - mean_local).pow(2).mean(dim=-1, keepdim=True)

            # priority bump = deterministic formula
            bump = (mean_local * 0.15) - (var_local * 0.05)

            # apply bump on diagonal-like positions
            # simulate "aging" effect
            x = x + bump * (0.3 + 0.1 * i)

            # L2-like normalization per iteration
            norm = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
            x = x / norm.clamp(min=1.0)

        return x


class InteractionRefiner(nn.Module):
    """
    Deterministic simulation of pairwise interactions WITHOUT attention.

    Replaces:
        for i in processes:
            for j in processes:
                interact(i, j)

    Using:
        - global mean
        - global variance
        - per-process deviation
        - reduction-only interactions
    """
    def __init__(self, state_dim: int, steps: int = 2):
        super().__init__()
        self.steps = steps
        self.state_dim = state_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, N, D]
        """
        x = feats

        for k in range(self.steps):
            global_mean = x.mean(dim=1, keepdim=True)     # [B,1,D]
            global_var = x.var(dim=1, keepdim=True)       # [B,1,D]

            # deviation from global
            dev = x - global_mean                          # [B,N,D]

            # deterministic interaction update
            upd = dev * 0.1 + global_var * 0.05

            # accumulation
            x = x + upd

            # small normalization
            norm = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
            x = x / norm.clamp(min=1.0)

        return x


class LoopConvergenceRefiner(nn.Module):
    """
    Deterministic loop-unrolled refinement.
    Replaces while(not converged) loops with a fixed number of layers.
    """

    def __init__(self, steps: int = 8):
        super().__init__()
        self.steps = steps

    def forward(self, scores: torch.Tensor,
                      proc_feats: torch.Tensor,
                      sys_load: torch.Tensor):

        """
        scores:     [B, N]
        proc_feats: [B, N, D]
        sys_load:   [B, 32]
        """

        B, N = scores.shape
        state = scores  # [B, N]

        # Expand system load to match process dimension
        sys_ctx = sys_load.unsqueeze(1).expand(-1, N, -1)  # [B, N, 32]

        # Loop-unrolled deterministic steps
        for i in range(self.steps):

            # Example deterministic update:
            # lightweight function of process features + system load
            delta = (
                proc_feats[..., 0] * 0.05 +           # priority-like feature
                proc_feats[..., 1] * 0.03 +           # burst/age-like feature
                sys_ctx[..., 0] * 0.01                # overall system pressure
            )  # [B, N]

            # Integrate deterministic update
            state = state + delta

            # optional early stop (deterministic, no ML)
            if i >= 4:
                if torch.all(state.mean(dim=1) > 0.9):
                    break

        return state


# ============================================================
#  CHUNK 4 — Load-Balancing Refiner + Final Top-K Selector
# ============================================================

class LoadBalancingRefiner(nn.Module):
    """
    Deterministic load-balancing step.

    Simulates:
        for each process:
            adjust score based on core pressure / imbalance

    Uses only:
        - reductions over core_states
        - simple arithmetic
        - per-batch broadcast
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                proc_scores: torch.Tensor,
                core_states: torch.Tensor) -> torch.Tensor:
        """
        proc_scores: [B, N]        (scalar score per process)
        core_states: [B, C, Dcore] (per-core state)
        """
        B, N = proc_scores.shape
        B2, C, Dcore = core_states.shape
        assert B == B2, "Batch dim mismatch between process scores and core states"

        # Per-core load = mean over core feature dim
        core_load = core_states.mean(dim=-1)        # [B, C]

        # Global core stats per batch
        core_mean = core_load.mean(dim=-1, keepdim=True)   # [B, 1]
        core_var  = core_load.var(dim=-1, keepdim=True)    # [B, 1]

        # Broadcast to [B, N]
        core_mean_exp = core_mean.expand(-1, N)     # [B, N]
        core_var_exp  = core_var.expand(-1, N)      # [B, N]

        # Deterministic adjustment:
        #   - If cores are heavily loaded (high mean), scores are dampened.
        #   - If imbalance/variance is high, we add a small penalty.
        adj = (-0.05 * core_mean_exp) - (0.02 * core_var_exp)

        out = proc_scores + adj

        # Simple normalization: keep scores roughly in [-1, 1]
        max_abs = out.abs().max(dim=-1, keepdim=True).values + 1e-6
        out = out / max_abs.clamp(min=1.0)

        return out


class FinalPolicySelector(nn.Module):
    """
    FINAL deterministic top-K selector.
    No MLPs, no softmax — just pure top-k.

    Simulates:
        - select Ncores best processes for each system in the batch
    """
    def __init__(self, num_cores: int):
        super().__init__()
        self.num_cores = num_cores

    def forward(self, scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        scores: [B, N] final scalar score per process

        Returns:
            {
              "scheduled_indices": [B, K],
              "scheduled_scores":  [B, K]
            }
        """
        B, N = scores.shape
        K = min(self.num_cores, N)

        topk_scores, topk_idx = torch.topk(scores, K, dim=1)

        return {
            "scheduled_indices": topk_idx,
            "scheduled_scores": topk_scores,
        }
# ============================================================
#  CHUNK 5 — HybridDeterministicScheduler (Main Pipeline)
# ============================================================

class HybridDeterministicScheduler(nn.Module):
    """
    The full deterministic GPU scheduler pipeline.

    It connects all modules:
      1. DeterministicPolicyRouter          (branch selection)
      2. ParallelPolicyScorer               (parallel scoring)
      3. InteractionRefiner                 (loop→layers)
      4. MLFQRefiner                        (loop→layers)
      5. LoadBalancingRefiner               (core interaction)
      6. LoopConvergenceRefiner             (while→layers)
      7. FinalPolicySelector                (top-k)
    """

    def __init__(self,
                 state_dim: int = 256,
                 num_policies: int = 5,
                 num_cores: int = 16):
        super().__init__()

        # ========= STAGE 1 =========
        self.router = DeterministicPolicyRouter(
            state_dim=state_dim,
            num_policies=num_policies
        )

        # ========= STAGE 2 =========
        self.policy_eval = ParallelPolicyScorer(
            state_dim=state_dim,
            num_policies=num_policies
        )

        # ========= STAGE 3 =========
        self.inter_refiner = InteractionRefiner(state_dim=state_dim)


        # ========= STAGE 4 =========
        self.mlfq_refiner = MLFQRefiner(
            state_dim=state_dim
        )

        # ========= STAGE 5 =========
        self.load_refiner = LoadBalancingRefiner()

        # ========= STAGE 6 =========
        self.loop_refiner = LoopConvergenceRefiner(steps=8)


        # ========= STAGE 7 =========
        self.final_selector = FinalPolicySelector(
            num_cores=num_cores
        )


    @torch.no_grad()
    def forward(self, system_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        system_state:
            {
              "proc_states": [B, N, D],
              "core_states": [B, C, 16],
              "system_load": [B, 32]
            }
        """

        proc_feats  = system_state["proc_states"]        # [B, N, D]
        core_states = system_state["core_states"]        # [B, C, Dcore]
        sys_load    = system_state["system_load"]        # [B, 32]

        # ========================================================
        # 1. ROUTING (deterministic policy picking)
        # ========================================================
        selected_policy, _ = self.router(proc_feats)
         # [B, N]

        # ========================================================
        # 2. PARALLEL SCORING
        # ========================================================
        raw_scores = self.policy_eval(proc_feats)         # [B, N, P]

        # pick the selected policy's score for each process
        B, N, P = raw_scores.shape
        chosen_score = raw_scores.gather(                  # [B, N]
            dim=2,
            index=selected_policy.unsqueeze(-1)
        ).squeeze(-1)

        # ========================================================
        # 3. INTERACTION REFINEMENT (loop→layers)
        # ========================================================
        inter_refined = self.inter_refiner(proc_feats)    # [B, N, D]

        # ========================================================
        # 4. MLFQ REFINEMENT (loop→layers)
        # ========================================================
        mlfq_refined = self.mlfq_refiner(proc_feats)      # [B, N, D]

        # Merge refinements into base scalar
        base = chosen_score + \
               0.15 * inter_refined.mean(dim=-1) + \
               0.10 * mlfq_refined.mean(dim=-1)           # [B, N]

        # ========================================================
        # 5. LOAD-BALANCE REFINEMENT (core interaction)
        # ========================================================
        lb = self.load_refiner(base, core_states)         # [B, N]

        # ========================================================
        # 6. LOOP CONVERGENCE REFINEMENT (while-loop→layers)
        # ========================================================
        final_scores = self.loop_refiner(lb, proc_feats, sys_load)   # [B, N]

        # ========================================================
        # 7. FINAL TOP-K SELECTION
        # ========================================================
        selection = self.final_selector(final_scores)

        return {
            "selected_policies": selected_policy,             # [B, N]
            "final_scores": final_scores,                     # [B, N]
            "scheduled_indices": selection["scheduled_indices"],  # [B, K]
            "scheduled_scores": selection["scheduled_scores"],    # [B, K]
        }
# ============================================================
#  CHUNK 6 — Simulation, Benchmark, CLI
# ============================================================

def make_system_state(
    batch_size: int,
    num_processes: int,
    state_dim: int,
    num_cores: int,
) -> dict:
    """
    Allocate the full system state in VRAM:

      proc_states : [B, N, D]  per-process feature state
      core_states : [B, C, 16] per-core state
      system_load : [B, 32]    global load / telemetry vector
    """
    proc_states = torch.randn(batch_size, num_processes, state_dim, device=DEVICE)
    core_states = torch.randn(batch_size, num_cores, 16, device=DEVICE)
    system_load = torch.randn(batch_size, 32, device=DEVICE)

    return {
        "proc_states": proc_states,
        "core_states": core_states,
        "system_load": system_load,
    }


class GPUSchedulerSimulation:
    """
    Wraps the HybridDeterministicScheduler and drives a synthetic workload.

    All state lives in VRAM, no CPU queues:
      - system_state is a dict of tensors on DEVICE
      - each step:
          * small deterministic-ish evolution of proc/system_load
          * scheduler is run once (one 'tick')
    """

    def __init__(
        self,
        batch_size: int = 512,
        num_processes: int = 128,
        state_dim: int = 256,
        num_cores: int = 16,
    ):
        self.batch_size   = batch_size
        self.num_processes = num_processes
        self.state_dim    = state_dim
        self.num_cores    = num_cores

        # Allocate state in VRAM
        self.state = make_system_state(
            batch_size=batch_size,
            num_processes=num_processes,
            state_dim=state_dim,
            num_cores=num_cores,
        )

        # The deterministic GPU scheduler pipeline
        self.scheduler = HybridDeterministicScheduler(
            state_dim=state_dim,
            num_policies=5,
            num_cores=num_cores,
        ).to(DEVICE)
        self.scheduler.eval()

    @torch.no_grad()
    def step(self, t: int) -> dict:
        """
        One scheduling tick:
          - evolve process/system state slightly (deterministic-ish noise)
          - call the scheduler
          - return scheduler outputs
        """

        # Small synthetic evolution of system state to avoid degenerate constant inputs
        # This mimics "time passing" without changing the architecture logic.
        proc = self.state["proc_states"]
        sysl = self.state["system_load"]

        # Very small noise scaled by a deterministic scalar per step
        step_scale = 0.01 * (1.0 + (t % 7) * 0.1)
        proc.add_(step_scale * torch.tanh(torch.sin(proc)))
        sysl.add_(step_scale * torch.tanh(sysl))

        self.state["proc_states"] = proc
        self.state["system_load"] = sysl

        # Run the deterministic scheduler
        out = self.scheduler(self.state)
        return out

    @torch.no_grad()
    def run(self, steps: int = 20) -> tuple:
        """
        Run the simulation for `steps` ticks and record timings (ms).
        Returns:
            (timings_list_ms, last_output_dict)
        """
        timings = []
        last_out = None

        for t in range(steps):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            last_out = self.step(t)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            dt_ms = (t1 - t0) * 1000.0
            timings.append(dt_ms)

            # Periodic logging
            if t % max(1, steps // 10) == 0:
                avg_ms = sum(timings) / len(timings)
                print(f"[step {t:3d}] avg = {avg_ms:.3f} ms")

        return timings, last_out


def benchmark_scheduler(
    batch_size: int = 512,
    num_processes: int = 128,
    state_dim: int = 256,
    num_cores: int = 16,
    steps: int = 20,
):
    """
    Construct a GPUSchedulerSimulation and benchmark its per-step runtime.
    """
    sim = GPUSchedulerSimulation(
        batch_size=batch_size,
        num_processes=num_processes,
        state_dim=state_dim,
        num_cores=num_cores,
    )

    timings, last_out = sim.run(steps)

    total_ms = sum(timings)
    avg_ms   = total_ms / max(1, len(timings))
    throughput = (steps * batch_size) / (total_ms / 1000.0) if total_ms > 0 else 0.0

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Avg Step Time : {avg_ms:.3f} ms")
    print(f"Total Runtime : {total_ms:.2f} ms")
    print(f"Throughput    : {throughput:.2f} schedulings/sec")
    print("========================================")

    return {
        "timings": timings,
        "last_output": last_out,
        "avg_ms": avg_ms,
        "throughput": throughput,
    }


# ============================================================
#  CLI ENTRY POINT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512, help="Batch size (B)")
    parser.add_argument("--processes", type=int, default=128, help="Processes per batch element (N)")
    parser.add_argument("--state_dim", type=int, default=256, help="Per-process state dimension (D)")
    parser.add_argument("--cores", type=int, default=16, help="Number of GPU 'cores' (C)")
    parser.add_argument("--steps", type=int, default=20, help="Number of scheduler ticks to run")
    args = parser.parse_args()

    print("==============================================")
    print(" DET. GPU HYBRID SCHEDULER  (ARCHITECTURE-B) ")
    print("==============================================")
    print(f"Device         : {DEVICE}")
    print(f"Batch Size     : {args.batch}")
    print(f"Processes/Task : {args.processes}")
    print(f"State Dim      : {args.state_dim}")
    print(f"GPU Cores      : {args.cores}")
    print(f"Steps          : {args.steps}")
    print("==============================================\n")

    _ = benchmark_scheduler(
        batch_size=args.batch,
        num_processes=args.processes,
        state_dim=args.state_dim,
        num_cores=args.cores,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
