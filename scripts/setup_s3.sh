#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Waiting for S3 to become available..."

# Loop until the 's3 ls' command succeeds
until aws --endpoint-url=http://localstack:4566 s3 ls &> /dev/null
do
  >&2 echo "S3 is unavailable - sleeping"
  sleep 1
done

>&2 echo "S3 is up - executing command"

# Create the bucket. The '|| true' part ensures that the script doesn't fail
# if the bucket already exists.
aws --endpoint-url=http://localstack:4566 s3 mb s3://credit-approval-data || true

echo "Bucket 'credit-approval-data' is ready."