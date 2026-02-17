"""
aws_deploy.py
Deploy the Hybrid Recommendation System to AWS SageMaker.
Runs: S3 upload -> Processing Job -> Training Job -> (optional) Endpoint

Prerequisites:
  1. AWS CLI configured: aws configure
  2. SageMaker execution role with S3 and SageMaker permissions
  3. S3 bucket created
  4. pip install boto3 sagemaker

Usage:
  # Full pipeline on AWS:
  python aws_deploy.py --bucket my-bucket-name --role arn:aws:iam::123456789:role/SageMakerRole

  # Only upload data (no training):
  python aws_deploy.py --bucket my-bucket-name --role <role_arn> --upload-only

  # Deploy inference endpoint (after training):
  python aws_deploy.py --bucket my-bucket-name --role <role_arn> --deploy-endpoint --model-uri s3://...
"""

import argparse
import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
import os


def upload_raw_data(bucket: str, s3_client):
    """Upload local data/raw/ CSVs to S3."""
    print("\n[1] Uploading raw data to S3...")
    for fname in ["products.csv", "users.csv", "transactions.csv"]:
        local = f"data/raw/{fname}"
        key = f"data/raw/{fname}"
        s3_client.upload_file(local, bucket, key)
        print(f"   Uploaded: s3://{bucket}/{key}")
    print("   Done.")


def run_processing_job(bucket: str, role: str, sess):
    """Launch SageMaker Processing Job for feature engineering."""
    print("\n[2] Starting SageMaker Processing Job...")
    processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name="hybrid-rec-preprocessing",
        sagemaker_session=sess,
    )
    processor.run(
        code="scripts/preprocessing.py",
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/data/raw/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/data/processed/",
            )
        ],
        wait=True,
        logs=True,
    )
    print(f"   Processing complete! Output: s3://{bucket}/data/processed/")


def run_training_job(bucket: str, role: str, sess, n_clusters: int = 4):
    """Launch SageMaker Training Job for KMeans."""
    print("\n[3] Starting SageMaker Training Job (KMeans)...")
    estimator = SKLearn(
        entry_point="train_kmeans.py",
        source_dir="scripts/",
        framework_version="1.0-1",
        instance_type="ml.t3.medium",
        instance_count=1,
        role=role,
        hyperparameters={
            "n_clusters": n_clusters,
            "random_state": 42,
            "max_iter": 300,
        },
        base_job_name="hybrid-rec-kmeans",
        sagemaker_session=sess,
    )
    estimator.fit(
        {"train": f"s3://{bucket}/data/processed/"},
        wait=True,
    )
    print(f"   Training complete! Model: {estimator.model_data}")
    return estimator.model_data


def deploy_endpoint(model_uri: str, role: str, sess):
    """Deploy KMeans model as a SageMaker real-time endpoint.
    WARNING: Endpoints incur charges. Delete immediately after testing!
    """
    from sagemaker.sklearn.model import SKLearnModel

    print("\n[4] Deploying SageMaker Endpoint...")
    print("   WARNING: ml.t2.medium instances are NOT free tier — delete after testing!")

    model = SKLearnModel(
        model_data=model_uri,
        role=role,
        entry_point="inference.py",
        source_dir="scripts/",
        framework_version="1.0-1",
        sagemaker_session=sess,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name="hybrid-rec-endpoint",
    )
    print(f"   Endpoint deployed: {predictor.endpoint_name}")

    # Quick test
    import json
    test_payload = {"features": [1200.0, 15, 80.0, 12, 4, 0.10]}
    response = predictor.predict(
        json.dumps(test_payload),
        initial_args={"ContentType": "application/json"},
    )
    print(f"   Test response: {response}")
    print("\n   IMPORTANT: Delete endpoint to stop charges:")
    print(f"   predictor.delete_endpoint()")
    print(f"   Or via CLI: aws sagemaker delete-endpoint --endpoint-name hybrid-rec-endpoint")
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Deploy Hybrid Rec System to AWS SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--role", required=True, help="SageMaker IAM role ARN")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--upload-only", action="store_true", help="Only upload data, skip training")
    parser.add_argument("--deploy-endpoint", action="store_true", help="Deploy inference endpoint")
    parser.add_argument("--model-uri", type=str, help="S3 URI for model.tar.gz (for endpoint deploy)")
    args = parser.parse_args()

    boto3.setup_default_session(region_name=args.region)
    s3 = boto3.client("s3")
    sess = sagemaker.Session()

    upload_raw_data(args.bucket, s3)

    if not args.upload_only:
        run_processing_job(args.bucket, args.role, sess)
        model_uri = run_training_job(args.bucket, args.role, sess, args.n_clusters)
        print(f"\nModel artifacts URI: {model_uri}")

    if args.deploy_endpoint:
        model_uri = args.model_uri
        if not model_uri:
            print("ERROR: --model-uri required for endpoint deployment")
            return
        deploy_endpoint(model_uri, args.role, sess)

    print("\n[Done] AWS deployment complete.")


if __name__ == "__main__":
    main()
