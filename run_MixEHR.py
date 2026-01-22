#!/usr/bin/env python3
"""
Command-line interface for running the MixEHR-SAGE pipeline.

Usage:
    python run_MixEHR.py <data_path> [options]

Example:
    python run_MixEHR.py ./data/ --output ./results/ --epochs 5 --batch-size 1000
"""
import subprocess
import sys
import os
import argparse

# Calculate the root directory once at module level
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None):
    """Run a subprocess command and raise on error."""
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    
    # If running from a subdirectory, add the parent directory to PYTHONPATH
    # so that modules in the root directory can be imported
    if cwd is not None:
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{ROOT_DIR}{os.pathsep}{current_pythonpath}"
        else:
            env['PYTHONPATH'] = ROOT_DIR
    
    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run MixEHR-SAGE pipeline on EHR data with dynamic modality support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (data in ./data/, output in ./results/)
    python run_MixEHR.py ./data/

    # Run with custom output directory and epochs
    python run_MixEHR.py ./data/ --output ./my_results/ --epochs 10

    # Run with custom batch size
    python run_MixEHR.py ./data/ --batch-size 500

Data Directory Requirements:
    The data directory must contain:
    - ukbb_metadata.csv: A CSV file with columns 'index', 'path', 'word_column'
      where 'index' is the modality name (first one is the guided modality),
      'path' is the path to the modality data file, and 'word_column' is the
      column name containing the words/codes.
    - Data files referenced in ukbb_metadata.csv

    The first modality listed in ukbb_metadata.csv is treated as the guided
    modality (typically ICD codes with PheCode mappings).
        """
    )
    parser.add_argument(
        'data_path',
        help='Path to the directory containing EHR data and ukbb_metadata.csv'
    )
    parser.add_argument(
        '--output', '-o',
        default='./results/',
        help='Directory to store model outputs (default: ./results/)'
    )
    parser.add_argument(
        '--store', '-s',
        default='./store/',
        help='Directory to store processed corpus (default: ./store/)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='Maximum number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size for training (default: 1000)'
    )
    parser.add_argument(
        '--max-docs', '-n',
        type=int,
        default=None,
        help='Maximum number of documents to process (default: all)'
    )
    parser.add_argument(
        '--skip-corpus',
        action='store_true',
        help='Skip corpus processing step (use existing corpus)'
    )
    parser.add_argument(
        '--skip-prior',
        action='store_true',
        help='Skip prior computation step (use existing priors)'
    )

    args = parser.parse_args()

    # Use the same Python interpreter
    python = sys.executable
    # Suppress Python warnings
    common_flags = ["-W", "ignore"]

    # Validate data path
    if not os.path.isdir(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist or is not a directory.")
        sys.exit(1)

    metadata_path = os.path.join(args.data_path, 'ukbb_metadata.csv')
    if not os.path.isfile(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        print("The data directory must contain 'ukbb_metadata.csv' with columns: index, path, word_column")
        sys.exit(1)

    # Create output directories if they don't exist
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.store, exist_ok=True)

    # 1) Process corpus
    if not args.skip_corpus:
        corpus_cmd = [python] + common_flags + [
            "corpus.py", "process",
            args.data_path, args.store
        ]
        if args.max_docs is not None:
            corpus_cmd.insert(-2, "-n")
            corpus_cmd.insert(-2, str(args.max_docs))
        run(corpus_cmd)
    else:
        print("Skipping corpus processing step...")

    # 2) Build priors in guide_prior
    if not args.skip_prior:
        guide_dir = os.path.join(os.path.dirname(__file__), "guide_prior")
        for script in ["get_doc_phecode.py", "get_prior_GMM.py", "get_token_counts.py"]:
            run([python] + common_flags + [script], cwd=guide_dir)
    else:
        print("Skipping prior computation step...")

    # 3) Fit the model
    run([python] + common_flags + [
        "main.py",
        args.store,
        args.output,
        "-epoch", str(args.epochs),
        "-batch_size", str(args.batch_size)
    ])

    print(f"\nPipeline completed successfully!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
