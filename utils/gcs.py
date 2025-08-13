import os
import json
from google.cloud import storage
import pandas as pd
from io import StringIO


def load_from_gcs(bucket_name, blob_name):
    """
    Downloads a CSV file from GCS and loads it into a Python variable.

    Args:
        bucket_name (str): GCS bucket name.
        blob_name (str): Path to the CSV file in GCS.

    Returns:
        dict or list: Parsed JSON content.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    json_content = blob.download_as_text()
    df = pd.read_csv(StringIO(json_content))
    return df.to_dict(orient="records")


def upload_to_gcs(bucket_name, data, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage and deletes the local file afterward.

    Args:
        bucket_name (str): GCS bucket name.
        data (dict): Data to be uploaded.
        destination_blob_name (str): Path to the file in GCS.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Convert data to JSON string
    json_data = json.dumps(data)

    # Upload JSON string directly
    blob.upload_from_string(json_data, content_type="application/json")


def upload_and_delete_local_file(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage and deletes the local file afterward."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file to GCS
    blob.upload_from_filename(source_file_name)

    # Delete the local file
    if os.path.exists(source_file_name):
        os.remove(source_file_name)
    else:
        pass


def list_from_gcs(bucket_name, folder_prefix, filename=None):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_prefix)

    if filename:
        return [blob.name for blob in blobs if filename in blob.name]
    else:
        return [blob.name for blob in blobs]
