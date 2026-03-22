import lancedb
import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
# Using lru_cache ensures we don't reconnect to S3 on every single request
@lru_cache()
def get_vector_table():
    s3_uri = "s3://your-bucket-name/lancedb_base"
    
    storage_options = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1")
    }
    
    # Connect to S3
    db = lancedb.connect(s3_uri, storage_options=storage_options)
    
    # Return the table (create it if it doesn't exist, or just open it)
    return db.open_table("pdf_embeddings")