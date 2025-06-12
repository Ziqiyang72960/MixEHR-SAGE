import subprocess
import sys
import os

def run(cmd, cwd=None):
    """Run a subprocess command and raise on error."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=cwd,
        check=True
    )

def main():
    # Use the same Python interpreter
    python = sys.executable
    # Suppress Python warnings
    common_flags = ["-W", "ignore"]

    # 1) Process corpus
    run([python] + common_flags + [
        "corpus.py", "process",
        "-n", "150",
        "./data/", "./store/"
    ])

    # 2) Build priors in guide_prior
    guide_dir = os.path.join(os.path.dirname(__file__), "guide_prior")
    for script in ["get_doc_phecode.py", "get_prior_GMM.py", "get_token_counts.py"]:
        run([python] + common_flags + [script], cwd=guide_dir)

    # 3) Fit the model
    run([python] + common_flags + ["main.py"])

if __name__ == "__main__":
    main()

