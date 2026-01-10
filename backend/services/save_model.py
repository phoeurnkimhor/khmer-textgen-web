from io import BufferedReader
from tusclient import client

def allowed_file(filename: str):
    return filename.lower().endswith((".pt", ".pth"))

def upload_file(bucket_name: str, file_name: str, file: BufferedReader, access_token: str, supabase_url: str):
    """
    Uploads a file to Supabase Storage using the Tus resumable API.
    Only .pt or .pth files are allowed.
    """
    if not allowed_file(file_name):
        raise ValueError("Invalid file type. Only .pt and .pth files are allowed.")

    tus_client = client.TusClient(
        f"{supabase_url}/storage/v1/upload/resumable",
        headers={
            "Authorization": f"Bearer {access_token}",  
            "x-upsert": "true"
        }
    )

    uploader = tus_client.uploader(
        file_stream=file,
        chunk_size=(6 * 1024 * 1024),  # 6 MB per chunk
        metadata={
            "bucketName": bucket_name,
            "objectName": file_name,
            "contentType": "application/octet-stream",
            "cacheControl": "3600",
        },
    )

    uploader.upload()
