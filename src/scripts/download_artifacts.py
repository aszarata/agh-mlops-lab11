import boto3

def download_from_s3(settings):
    s3 = boto3.client('s3')

    s3.download_file(
        settings.s3_bucket_name,
        settings.s3_classifier_path,
        str(settings.classifier_joblib_path)
    )
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=settings.s3_bucket_name, Prefix=settings.s3_sentence_transformer_path):
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_path = settings.model_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(settings.s3_bucket_name, key, str(local_path))

if __name__ == "__main__":
    download_from_s3()