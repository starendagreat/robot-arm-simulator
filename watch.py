"""
Watch a CLAW-GRASP policy act in the RENDERED VPython scene, so you can see the progress.

A browser window opens showing the real simulation — one pre-aimed arm and a box moving toward
the claw — with the claw's commit driven by the trained policy, so you see exactly what the
agent has learned. It's the same env used for training, just loaded with real VPython and paced
to ~real time. Each episode is one grasp attempt (the box travels in; the policy times a close).

Usage
-----
  python watch.py                 # load sac_claw.zip and watch it
  python watch.py --stage 4       # watch on a specific curriculum stage's distribution
  python watch.py --algo ppo      # watch a PPO model instead
  python watch.py --random        # no model needed — watch random actions (test the view)
  python watch.py --slow 2        # run at half speed (bigger = slower), to study a grasp

Stop with Ctrl-C (or close the browser tab).
"""
import argparse
import os

from claw_env import ClawGraspEnv
from train import resolve_env_kwargs

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--stage", type=int, default=None,
                    help="curriculum stage to watch on (1-7); omit for the base defaults")
    ap.add_argument("--random", action="store_true",
                    help="watch random actions instead of a trained model")
    ap.add_argument("--slow", type=float, default=1.0,
                    help="slow-motion factor (2 = half speed); helps study a grasp")
    args = ap.parse_args()

    env_kwargs = resolve_env_kwargs(args.stage)
    env = ClawGraspEnv(render=True, **env_kwargs)     # real VPython window
    if args.slow and args.slow > 1.0:
        env.dt = env.dt * float(args.slow)            # rate() paces off dt -> slower playback

    if args.random:
        def predict(_obs):
            return env.action_space.sample()
        print("watching RANDOM actions (no model). Ctrl-C to stop.")
    else:
        from stable_baselines3 import SAC, PPO
        path = os.path.join(_HERE, f"{args.algo}_claw.zip")
        if not os.path.exists(path):
            raise SystemExit(f"no model at {path} — run `python train.py` first, "
                             f"or use `python watch.py --random`.")
        model = {"sac": SAC, "ppo": PPO}[args.algo].load(path)

        def predict(obs):
            return model.predict(obs, deterministic=True)[0]
        print(f"watching {args.algo.upper()} ({os.path.basename(path)}). Ctrl-C to stop.")

    obs, _ = env.reset()
    ep = grabbed = 0
    try:
        while True:
            obs, _r, term, trunc, info = env.step(predict(obs))
            if term or trunc:
                ep += 1
                got = info.get("caught", False)
                grabbed += int(got)
                status = (f"GRABBED (centering {info.get('centering', 0.0):.2f})" if got
                          else f"missed  ({info.get('outcome', '')})")
                print(f"episode {ep:3d}: {status:32s}  "
                      f"(grab rate {grabbed}/{ep} = {grabbed / ep:.0%})")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
