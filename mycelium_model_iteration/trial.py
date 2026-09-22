import diffusion_verification as dv
import subprocess

dv.RNG_SEED = 8

subprocess.run(["python", "diffusion_verification.py"])

