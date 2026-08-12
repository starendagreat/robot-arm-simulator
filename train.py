"""
Train / evaluate a Stable-Baselines3 agent on the CLAW GRASP environment.

The env (claw_env.ClawGraspEnv) has a CONTINUOUS action space (a single commit scalar), so the
default algorithm is SAC — the sample-efficient, off-policy actor-critic that's usually the
easiest good choice for continuous control. PPO is also available (--algo ppo).

The agent's whole job is TIMING one grasp on a box moving toward a pre-aimed claw (see
claw_env.py). Mean episode reward tracks task success cleanly here (a grab is worth ~+1.1, a
miss < +0.3), so best-model-by-mean-reward is trustworthy — unlike the old catch+deliver task
where return and success had decoupled.

Usage
-----
  python train.py                       # RESUME the saved model (or start fresh if none)
  python train.py --timesteps 300000    # train this many MORE steps this session
  python train.py --fresh               # ignore the saved model and start from scratch
  python train.py --n-envs 8            # more parallel envs (up to your CPU cores)
  python train.py --n-envs 1            # single env (no multiprocessing)
  python train.py --algo ppo            # use PPO instead of SAC
  python train.py --eval                # load the saved model and watch its grab rate
  python train.py --eval --episodes 50  # evaluate over 50 episodes
  python train.py --eval --best         # evaluate the BEST checkpoint, not the rolling model
  python train.py --no-eval             # train without best-model checkpointing
  python train.py --lr 1e-4             # LOWER the learning rate for continued fine-tuning
                                        #   (avoids the over-training collapse past convergence)

Curriculum (see the STAGES table below):
  python train.py --stage 1 --fresh     # START the curriculum at stage 1 (from scratch)
  python train.py --stage 2             # advance: RESUME (warm-start) into stage 2 — NOT --fresh
  python train.py --eval --stage 2      # check stage 2's rolling model's grab rate
  python train.py --eval --stage 2 --best   # ...its best per-stage checkpoint
Advance a stage once its deterministic eval reliably solves it AND has plateaued — as a rule of
thumb: the fixed-approach stages 1-3 grab >= ~95%; the widening-angle stages 4-5 grab >= ~85%;
stage 6 (full randomisation) to your target; stage 7 (imperfect alignment) is the robustness
finisher — expect a somewhat lower grab ceiling since the box is always off-centre. If a widening
stage stalls, it usually means the
jump was still too big — narrow the gap to the previous stage rather than just training longer.
Resume (don't --fresh) between stages: the reward function is identical across them, so the
replay buffer stays valid and the warm-started policy is the whole point of the curriculum.

Training RESUMES from <algo>_claw.zip if it exists, so repeated sessions keep improving
the same model (use --fresh to start over). Progress is saved at the end AND on Ctrl-C, so
an interrupted run isn't lost. SAC also saves its replay buffer (<algo>_claw_buffer.pkl)
so a resume continues smoothly.

While training, a periodic DETERMINISTIC eval runs and writes the best-scoring model so far to
<algo>_best/best_model.zip — so off-policy over-training can't overwrite a good policy with a
degraded one. Load it with --eval --best (or point watch.py at it). IMPORTANT: whenever you
CHANGE the reward function, retrain with --fresh — resuming reuses the saved replay buffer,
whose stored rewards were computed under the OLD reward and will fight the new one.

The reward is a single TERMINAL score per episode (grab +1 with a centering bonus, or a
distance-graded miss), so credit assignment is clean; the curriculum ramps speed then approach
angle. This script is the training harness; the learning problem lives in claw_env.py.
"""
import argparse
import json
import math
import os
from functools import partial

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:                                       # SB3 >= 2.x
    from stable_baselines3.common.utils import FloatSchedule as _const_schedule
except ImportError:                        # older SB3
    from stable_baselines3.common.utils import get_schedule_fn as _const_schedule

from claw_env import ClawGraspEnv


# Reward-affecting env kwargs. If any of these change, a best score saved under the old settings
# is on a different scale and must NOT be compared to a new session's — so the running best is
# reset when they differ (otherwise a looser/tighter reward could spuriously "beat" the peak).
# These are the reward-FORMULA knobs only; the per-stage speed/angle differences are expected
# (each stage has its own best checkpoint), so they're deliberately excluded here.
_REWARD_KEYS = ("max_steps", "near_reward", "centering_weight", "centering_tol",
                "time_penalty", "catchable_dist", "commit_threshold", "grab_resolve_frames",
                "settle_frames", "frame_skip")


def _reward_sig(env_kwargs):
    """A signature of the reward-affecting settings, stored alongside the persisted best score
    so a reward change invalidates it."""
    return {k: env_kwargs.get(k) for k in _REWARD_KEYS}


class PersistentEvalCallback(EvalCallback):
    """EvalCallback whose 'best' survives ACROSS training sessions.

    SB3's EvalCallback resets its best score to -inf every run, so a fresh session's first eval
    overwrites best_model.zip even with a WORSE policy — which is exactly how a genuine
    cross-session peak got clobbered here (a later, degraded session's best replaced the real
    peak). We persist the best mean reward (+ the reward signature it was measured under) to a
    small JSON next to best_model.zip and reload it on start, so best_model.zip is only
    overwritten by a policy that genuinely beats the ALL-TIME best under the SAME reward. A
    reward change resets the running best (old scores aren't comparable)."""

    def __init__(self, *args, best_score_path=None, reward_sig=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_score_path = best_score_path
        self._reward_sig = reward_sig
        if best_score_path and os.path.exists(best_score_path):
            try:
                data = json.load(open(best_score_path))
                if data.get("reward_sig") == reward_sig:
                    self.best_mean_reward = float(data["best_mean_reward"])
                    print(f"  running-best mean reward carried over: {self.best_mean_reward:.3f} "
                          f"(best_model.zip only replaced if this is beaten)")
                else:
                    print("  reward settings changed since the saved best -> running best reset.")
            except Exception:
                pass                                   # corrupt/old file -> start fresh (-inf)

    def _on_step(self):
        prev = self.best_mean_reward
        keep_going = super()._on_step()                # runs the eval; may save a new best_model
        if self._best_score_path and self.best_mean_reward > prev:
            try:
                with open(self._best_score_path, "w") as f:
                    json.dump({"best_mean_reward": self.best_mean_reward,
                               "reward_sig": self._reward_sig}, f)
            except Exception:
                pass
        return keep_going

_ALGOS = {"sac": SAC, "ppo": PPO}
_HERE = os.path.dirname(os.path.abspath(__file__))

# Base env construction args (shared by every stage + eval). These are the reward-formula and
# geometry knobs the curriculum does NOT vary; each stage layers speed/angle/offset on top.
#   near_reward / centering_weight / centering_tol / time_penalty: the reward shape (see
#     claw_env.py).
#   centering_weight=0.5 (was 0.2): the claw is fixed all episode, so as the box drifts through,
#     ITS lateral offset from the grasp axes generally changes with WHERE along the path the
#     policy commits -- commit timing controls precision, not just depth. At 0.2 any grab (even
#     barely-centred) already captured ~83% of the max reward (1.0 of 1.0-1.2), too weak a pull
#     toward the precision-optimal commit moment. 0.5 widens the grab range to 1.0-1.5 (a 50%
#     swing) while a worst-case grab (1.0) still comfortably beats the best possible miss
#     (near_reward=0.3), so this only sharpens WHICH moment to commit at, never whether to.
#   frame_skip=2 (was 5): finer commit-timing resolution. Diagnosis found the box's velocity
#     component along the claw's OWN finger-closing axis varies by azimuth -- near head-on it's
#     almost entirely along that axis, making the correct commit window much narrower in TIME
#     there than at extreme azimuth (where velocity is mostly off-plane, not fighting the
#     fingers' closing motion). frame_skip=5 (0.05s/decision) was too coarse to reliably land
#     that narrow window; 2 (0.02s/decision) gives ~2.5x finer control right where it matters.
#   max_steps=150 (was 60): scaled up ~2.5x to match, so a step's finer time-resolution doesn't
#     shrink the real-time budget an episode covers (same reasoning as the earlier speed-floor
#     widening: same physical traversal, more but smaller steps).
ENV_KWARGS = dict(max_steps=150, frame_skip=2, near_reward=0.3, centering_weight=0.5,
                  centering_tol=0.12, time_penalty=0.002, catchable_dist=0.3,
                  grasp_radius=1.8, spawn_gap=0.9, grab_resolve_frames=14, settle_frames=6)

# ── Curriculum ────────────────────────────────────────────────────────────────────────────
# Each stage layers ONE new difficulty axis on top of ENV_KWARGS. Warm-start each stage from the
# previous one (resume, NOT --fresh — the reward function is identical across stages, so the
# replay buffer stays consistent). Advance once the deterministic eval (python train.py --eval
# --stage N) reliably solves it — see the docstring.
#
# The box always drifts WORLD −x (like the live sim); the curriculum ramps SPEED first, then
# widens the base-rotation ANGLE so the −x drift meets the claw at ever-larger side angles, and
# finally adds elevation + off-centre crossings. Only the FRONT hemisphere is used (±pi/2) since
# the sim's boxes come from +x and are intercepted in front:
#   1  −x drift head-on, slow (0.5)       -> learn to TIME the commit as the box crosses
#   2  head-on, speed 1.0                  -> a faster window
#   3  head-on, speed random 0.5-2.5       -> full speed range (must anticipate the close)
#   4  widen intercept angle to ±45deg     -> the box crosses at a moderate side angle
#   5  widen to ±90deg (full front hemi)   -> box crosses fully sideways
#   6  + elevation + lateral offset        -> full front-hemisphere randomisation + centering
#   7  IMPERFECT ALIGNMENT: box always off -> robustness to imperfect aim — the box crosses
#      centre (offset 0.08-0.16 in y-z)       noticeably off-centre but still inside the catch
#                                             area (~0.14 envelope), so it must grip a box that
#                                             is NOT lined up. Trains for the sim's real aim error
#                                             (tracker lag / analytic aim never centres perfectly).
#
# NOTE: a 0.2 slow-speed floor (+ a "stage 8" isolated slow-speed bridge) was tried and reverted —
# it made the combined stage-7 task WORSE, not better (repeated regressions across several
# retraining attempts; see project history). Speed floor is back to 0.5 everywhere below.
_HALF_PI = math.pi / 2
_QTR_PI = math.pi / 4

STAGES = {
    1: dict(speed_range=(0.5, 0.5), azimuth_range=(0.0, 0.0)),
    2: dict(speed_range=(1.0, 1.0), azimuth_range=(0.0, 0.0)),
    3: dict(speed_range=(0.5, 2.5), azimuth_range=(0.0, 0.0)),
    4: dict(speed_range=(0.5, 2.5), azimuth_range=(-_QTR_PI, _QTR_PI)),
    5: dict(speed_range=(0.5, 2.5), azimuth_range=(-_HALF_PI, _HALF_PI)),
    6: dict(speed_range=(0.5, 2.5), azimuth_range=(-_HALF_PI, _HALF_PI),
            elevation_range=(-0.3, 0.3), lateral_offset_max=0.12),
    7: dict(speed_range=(0.5, 2.5), azimuth_range=(-_HALF_PI, _HALF_PI),
            elevation_range=(-0.3, 0.3), lateral_offset_min=0.08, lateral_offset_max=0.16),
}


def resolve_env_kwargs(stage):
    """Merge the base ENV_KWARGS with the chosen curriculum stage's overrides (stage=None ->
    just the base). A plain dict so it pickles cleanly to SubprocVecEnv workers on Windows."""
    kw = dict(ENV_KWARGS)
    if stage is not None:
        kw.update(STAGES[stage])
    return kw


def _progress_bar_available():
    """SB3's progress bar needs tqdm + rich; use it only if both are installed."""
    try:
        import tqdm, rich  # noqa: F401
        return True
    except ImportError:
        return False


def make_env(env_kwargs):
    """A single Monitored env instance (Monitor records episode reward/length). Must be a
    top-level function so SubprocVecEnv can pickle it to each worker process on Windows —
    the stage kwargs travel via functools.partial(make_env, env_kwargs), which pickles fine
    (module-level func + a plain dict), unlike mutating a module global that a fresh worker
    import would reset to the default."""
    return Monitor(ClawGraspEnv(**env_kwargs))


def model_path(algo):
    # A CLAW-specific filename so the archived catch+deliver model (<algo>_robotarm.zip) is
    # never loaded or overwritten by this task (its obs/action dims differ anyway).
    return os.path.join(_HERE, f"{algo}_claw.zip")


def buffer_path(algo):
    return model_path(algo).replace(".zip", "_buffer.pkl")


def best_dir(algo, stage=None):
    """Folder EvalCallback writes the best-so-far model into (best_model.zip) plus its
    evaluation log (evaluations.npz). Kept separate from the rolling model_path so a
    degraded late-training session can never overwrite the best checkpoint. Made PER-STAGE
    (<algo>_stage{N}_best) so advancing the curriculum never clobbers an earlier stage's
    known-good policy — you can always roll back to it."""
    name = f"{algo}_claw_best" if stage is None else f"{algo}_claw_stage{stage}_best"
    return os.path.join(_HERE, name)


def best_model_path(algo, stage=None):
    return os.path.join(best_dir(algo, stage), "best_model.zip")


def train(algo, timesteps, n_envs, fresh=False, eval_freq=25_000, n_eval_episodes=5,
          use_eval=True, stage=None, lr=None):
    Algo = _ALGOS[algo]
    env_kwargs = resolve_env_kwargs(stage)
    if stage is not None:
        print(f"curriculum stage {stage}: {STAGES[stage]}")
    # Collect experience from n_envs env copies in parallel (separate processes) to speed up
    # data collection ~linearly with CPU cores. A single env -> a plain in-process DummyVecEnv.
    # partial(make_env, env_kwargs) carries the stage config into each (possibly subprocess) env.
    if n_envs > 1:
        env = SubprocVecEnv([partial(make_env, env_kwargs) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([partial(make_env, env_kwargs)])

    path = model_path(algo)
    resume = (not fresh) and os.path.exists(path)
    if resume:
        # CONTINUE the existing model instead of starting over — this is why a new session
        # used to reset ep_rew_mean: it was building a fresh model every time.
        print(f"resuming from {path} ...")
        model = Algo.load(path, env=env)
        if lr is not None:
            # Override the saved learning rate for continued fine-tuning. SAC restores the LR
            # it was trained with, so without this every resume keeps the original 3e-4 — too
            # large past convergence, which over-sharpens the policy onto the easy region
            # (the over-training collapse). Set both the value and the schedule SB3 reads each
            # update; it persists into the next save, so the lower LR sticks.
            model.learning_rate = lr
            model.lr_schedule = _const_schedule(lr)
            print(f"  learning rate overridden -> {lr:g}")
        if algo == "sac" and os.path.exists(buffer_path(algo)):
            # SAC's replay buffer is hard-wired to the env COUNT it was saved with, so it
            # only loads if this run uses the same --n-envs. Keep the fresh (correctly-sized)
            # empty buffer so we can fall back to it if the saved one is incompatible — the
            # MODEL is unaffected either way, only the auxiliary experience memory.
            fresh_buffer = model.replay_buffer
            model.load_replay_buffer(buffer_path(algo))
            if model.replay_buffer.n_envs != env.num_envs:
                print(f"  saved replay buffer is for {model.replay_buffer.n_envs} envs but "
                      f"this run uses {env.num_envs} — discarding it (the model is kept; the "
                      f"buffer will refill). Use --n-envs {model.replay_buffer.n_envs} to reuse it.")
                model.replay_buffer = fresh_buffer
            else:
                print(f"  reloaded replay buffer ({model.replay_buffer.size()} transitions)")
    else:
        if fresh and os.path.exists(path):
            print("(--fresh) ignoring the existing model; starting from scratch.")
        # Sensible starting hyperparameters; MlpPolicy = small fully-connected net over the
        # observation. Tune later once the reward/curriculum is sorted out.
        lr_init = lr if lr is not None else 3e-4      # --lr sets the initial LR for a fresh model
        if algo == "sac":
            model = Algo("MlpPolicy", env, verbose=1, learning_rate=lr_init,
                         buffer_size=200_000, batch_size=256, gamma=0.99, tau=0.005,
                         train_freq=1, gradient_steps=1)
        else:  # ppo
            model = Algo("MlpPolicy", env, verbose=1, learning_rate=lr_init,
                         n_steps=2048, batch_size=256, gamma=0.99, gae_lambda=0.95)

    def save():
        model.save(path)
        if algo == "sac":
            model.save_replay_buffer(buffer_path(algo))
        print(f"saved model to {path}")

    # Best-model checkpointing: periodically run a DETERMINISTIC eval on a separate env and
    # save the model ONLY when its mean episode reward beats the best seen so far. This is the
    # safety net against off-policy over-training degradation — the rolling model_path can
    # still drift down and get overwritten each session, but best_model.zip is frozen at the
    # peak, so a bad late session can never clobber your good policy. (Mean reward tracks the
    # true goal here: a grab is the only large reward term, so a model that stops grabbing earns
    # less and won't be saved as "best".) Runs in-process on its own env.
    # PersistentEvalCallback carries the best score ACROSS sessions (see its docstring) — plain
    # EvalCallback resets it every run, which let a degraded later session overwrite the peak.
    eval_callback = None
    if use_eval:
        # Dedicated in-process eval env, built with the SAME stage config as training.
        eval_env = DummyVecEnv([partial(make_env, env_kwargs)])
        # EvalCallback counts eval_freq in CALLBACK calls, and one call == one vectorised env
        # step == n_envs timesteps, so divide to eval roughly every `eval_freq` TOTAL steps.
        eval_callback = PersistentEvalCallback(
            eval_env,
            best_model_save_path=best_dir(algo, stage),
            log_path=best_dir(algo, stage),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            best_score_path=os.path.join(best_dir(algo, stage), "best_score.json"),
            reward_sig=_reward_sig(env_kwargs),
        )

    print(f"training {algo.upper()} for {timesteps} more timesteps across {n_envs} env(s)"
          f"{' (resumed)' if resume else ''}...")
    if eval_callback is not None:
        print(f"  best-model checkpointing -> {best_model_path(algo, stage)} "
              f"(eval every ~{eval_freq} steps over {n_eval_episodes} episodes)")
    # log_interval = how often SB3 prints a training-stats block. For off-policy SAC it's
    # counted in EPISODES, so 20 = one printout every 20 episodes. (For PPO it's iterations.)
    # reset_num_timesteps=False when resuming so the step counter (and schedules) continue,
    # and `timesteps` is trained as ADDITIONAL steps on top of what the model already has.
    try:
        model.learn(total_timesteps=timesteps, log_interval=20,
                    reset_num_timesteps=not resume,
                    callback=eval_callback,
                    progress_bar=_progress_bar_available())
    except KeyboardInterrupt:
        print("\ninterrupted — saving progress before exit...")
    finally:
        save()                                            # always save (even on Ctrl-C)
        env.close()
        if eval_callback is not None:
            eval_env.close()


def evaluate(algo, episodes, use_best=False, stage=None):
    path = best_model_path(algo, stage) if use_best else model_path(algo)
    if not os.path.exists(path):
        which = "best-checkpoint" if use_best else "trained"
        raise SystemExit(f"no {which} model at {path} — run `python train.py` first"
                         f"{' (best_model.zip is only written once an eval runs)' if use_best else ''}.")
    print(f"evaluating {os.path.relpath(path, _HERE)}"
          f"{f' on curriculum stage {stage}' if stage is not None else ''} ...")
    model = _ALGOS[algo].load(path)
    env = ClawGraspEnv(**resolve_env_kwargs(stage))
    grabbed, returns, centerings = 0, [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc
        if info.get("caught", False):                    # the claw formed a grip this episode
            grabbed += 1
            centerings.append(info.get("centering", 0.0))
        returns.append(ep_ret)
    env.close()
    mean_cent = sum(centerings) / len(centerings) if centerings else 0.0
    print(f"\n{algo.upper()} over {episodes} episodes:")
    print(f"  grab rate      : {grabbed}/{episodes} = {grabbed / episodes:.0%}")
    print(f"  mean centering : {mean_cent:.2f}  (of grabs; 1.0 = dead-centre)")
    print(f"  mean return    : {sum(returns) / len(returns):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=list(_ALGOS), default="sac",
                    help="RL algorithm (default: sac)")
    ap.add_argument("--timesteps", type=int, default=200_000,
                    help="training steps (default: 200000)")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="parallel envs for data collection (default: 4; 1 = no subprocess)")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new model from scratch instead of resuming the saved one")
    ap.add_argument("--eval", action="store_true",
                    help="evaluate the saved model instead of training")
    ap.add_argument("--episodes", type=int, default=20,
                    help="episodes to evaluate over (default: 20)")
    ap.add_argument("--best", action="store_true",
                    help="with --eval, load the best-checkpoint model (<algo>_best/best_model.zip) "
                         "instead of the rolling one")
    ap.add_argument("--eval-freq", type=int, default=25_000,
                    help="train: run a best-model eval about every this many timesteps (default: 25000)")
    ap.add_argument("--n-eval-episodes", type=int, default=5,
                    help="train: episodes per best-model eval (default: 5)")
    ap.add_argument("--no-eval", action="store_true",
                    help="train: disable best-model checkpointing (no periodic eval)")
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), default=None,
                    help="curriculum stage (1-7); sets the box speed / approach-angle / offset "
                         "per the STAGES table. Omit for the base env (stage-1-like defaults). "
                         "Applies to both training and --eval, and picks the per-stage best "
                         "checkpoint.")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate (default: keep the model's saved LR, or 3e-4 "
                         "for a fresh model). Lower it (e.g. 1e-4) for continued fine-tuning past "
                         "convergence to avoid the over-training collapse. Persists into the save.")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.algo, args.episodes, use_best=args.best, stage=args.stage)
    else:
        train(args.algo, args.timesteps, args.n_envs, args.fresh,
              eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
              use_eval=not args.no_eval, stage=args.stage, lr=args.lr)


if __name__ == "__main__":
    main()
