#!/usr/bin/env python3
"""
Upload model weights to S3 bucket river-weights in us-east-1

Usage:
    python upload_weights.py                           # Upload best checkpoints for both models
    python upload_weights.py --model baseline          # Upload baseline only
    python upload_weights.py --model attention         # Upload attention only
    python upload_weights.py --checkpoint path/to/ckpt # Upload specific checkpoint
    python upload_weights.py --prefix my-experiment    # Custom S3 prefix
"""

import argparse
import os
import boto3
from datetime import datetime
from botocore.config import Config

BUCKET = "river-weights"
REGION = "us-east-1"

# Default checkpoint paths
CHECKPOINTS = {
    "baseline": "trained_models/baseline_rnn/checkpoint/best_checkpoint",
    "attention": "trained_models/rnn_attention/checkpoint/best_checkpoint",
}

# Also upload the args file alongside checkpoints
ARGS_FILES = {
    "baseline": "trained_models/baseline_rnn/checkpoint/args.yaml",
    "attention": "trained_models/rnn_attention/checkpoint/args.yaml",
}


def upload_file(s3_client, local_path, s3_key):
    """Upload a single file to S3"""
    if not os.path.exists(local_path):
        print(f"  Skipping {local_path} (not found)")
        return False

    file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
    print(f"  Uploading {local_path} ({file_size:.1f} MB) -> s3://{BUCKET}/{s3_key}")

    s3_client.upload_file(local_path, BUCKET, s3_key)
    print(f"  Done")
    return True


def upload_checkpoint(s3_client, model_name, checkpoint_path, args_path, prefix):
    """Upload a checkpoint and its args file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # S3 keys
    checkpoint_key = f"{prefix}/{model_name}/{timestamp}/best_checkpoint"
    args_key = f"{prefix}/{model_name}/{timestamp}/args.yaml"

    # Also create a "latest" symlink-like copy
    latest_checkpoint_key = f"{prefix}/{model_name}/latest/best_checkpoint"
    latest_args_key = f"{prefix}/{model_name}/latest/args.yaml"

    print(f"\nUploading {model_name} model:")

    # Upload checkpoint
    if upload_file(s3_client, checkpoint_path, checkpoint_key):
        # Copy to latest
        s3_client.copy_object(
            Bucket=BUCKET,
            CopySource=f"{BUCKET}/{checkpoint_key}",
            Key=latest_checkpoint_key
        )
        print(f"  Updated latest -> {latest_checkpoint_key}")

    # Upload args
    if upload_file(s3_client, args_path, args_key):
        s3_client.copy_object(
            Bucket=BUCKET,
            CopySource=f"{BUCKET}/{args_key}",
            Key=latest_args_key
        )
        print(f"  Updated latest -> {latest_args_key}")

    return checkpoint_key


def main():
    parser = argparse.ArgumentParser(description="Upload model weights to S3")
    parser.add_argument("--model", choices=["baseline", "attention", "both"], default="both",
                        help="Which model to upload (default: both)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint file to upload")
    parser.add_argument("--prefix", type=str, default="nejm-brain-to-text",
                        help="S3 prefix/folder (default: nejm-brain-to-text)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    # Initialize S3 client
    config = Config(region_name=REGION)
    s3_client = boto3.client("s3", config=config)

    print(f"Bucket: s3://{BUCKET}")
    print(f"Region: {REGION}")
    print(f"Prefix: {args.prefix}")

    if args.dry_run:
        print("\n[DRY RUN - no files will be uploaded]")

    # Handle specific checkpoint upload
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(args.checkpoint)
        s3_key = f"{args.prefix}/custom/{timestamp}/{filename}"

        if args.dry_run:
            print(f"\nWould upload: {args.checkpoint} -> s3://{BUCKET}/{s3_key}")
        else:
            upload_file(s3_client, args.checkpoint, s3_key)
        return 0

    # Upload model checkpoints
    models_to_upload = ["baseline", "attention"] if args.model == "both" else [args.model]

    for model in models_to_upload:
        checkpoint_path = CHECKPOINTS[model]
        args_path = ARGS_FILES[model]

        if args.dry_run:
            print(f"\nWould upload {model}:")
            print(f"  {checkpoint_path}")
            print(f"  {args_path}")
        else:
            upload_checkpoint(s3_client, model, checkpoint_path, args_path, args.prefix)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
