"""RustFS output paths, credentials and the local view of processed objects."""

from pathlib import Path, PurePosixPath
import json
import os
import re
import subprocess
import tempfile

from emovid_segments import split_s3

DEFAULT_OUTPUT_URI = "s3://personalized-fm/processed"
DEFAULT_MOUNT = Path("/home/zhangyn/rustfs")


def recording_path(raw_uri, raw_root="s3://oi-data/EMOVID_RAW"):
    """Use the outer acquisition directory requested by the user as the name."""
    bucket, key = split_s3(raw_uri)
    root_bucket, prefix = split_s3(raw_root)
    prefix = prefix.rstrip("/") + "/"
    if bucket != root_bucket or not key.startswith(prefix):
        raise ValueError("Raw recording is outside --raw-root")
    relative = PurePosixPath(key[len(prefix):])
    if any(part in (".", "..", "") for part in relative.parts) or relative.suffix.lower() not in (".bdf", ".edf"):
        raise ValueError("Unsafe or unsupported source recording path")
    return PurePosixPath(relative.parts[0])


def credentials_for(client):
    # Reuse the same locally authorized profile as the existing RustFS reader.
    # Values are passed in memory/environment only, never logged or serialized.
    return client._request_signer._credentials.get_frozen_credentials()


def lance_storage_options(client):
    credentials = credentials_for(client)
    options = {"aws_access_key_id": credentials.access_key, "aws_secret_access_key": credentials.secret_key,
               "aws_region": client.meta.region_name or "us-east-1", "aws_endpoint": client.meta.endpoint_url,
               "allow_http": str(client.meta.endpoint_url.startswith("http://")).lower(),
               "aws_virtual_hosted_style_request": "false", "aws_skip_signature": "false"}
    if credentials.token:
        options["aws_session_token"] = credentials.token
    return options


class OutputStore:
    def __init__(self, client, root=DEFAULT_OUTPUT_URI):
        self.client = client
        self.bucket, self.prefix = split_s3(root)
        self.prefix = self.prefix.rstrip("/")
        self.root = f"s3://{self.bucket}/{self.prefix}"
        self.options = lance_storage_options(client)

    def key(self, relative):
        path = PurePosixPath(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Output path must be relative and cannot traverse parents")
        return self.prefix + "/" + str(path)

    def uri(self, relative):
        return f"s3://{self.bucket}/{self.key(relative)}"

    def exists(self, relative):
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(relative))
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_json(self, relative):
        response = self.client.get_object(Bucket=self.bucket, Key=self.key(relative))
        try:
            return json.loads(response["Body"].read())
        finally:
            response["Body"].close()

    def put_json(self, relative, value):
        self.put(relative, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode(), "application/json")

    def put(self, relative, data, content_type="application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=self.key(relative), Body=data, ContentType=content_type)


def mount_processed(client, mountpoint=DEFAULT_MOUNT, output_uri=DEFAULT_OUTPUT_URI):
    """Mount the exact processed prefix, with no disk-backed waveform cache."""
    mountpoint = Path(mountpoint).absolute()
    bucket, prefix = split_s3(output_uri)
    expected = f"emovid_processed:{bucket}/{prefix.rstrip('/')}"
    if os.path.ismount(mountpoint):
        source = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(mountpoint)], text=True).strip()
        if re.sub(r"\{[^}]*\}(?=:)", "", source) != expected:
            raise ValueError(f"Mountpoint already maps a different source: {source}")
        return mountpoint
    mountpoint.mkdir(parents=True, exist_ok=True)
    if any(mountpoint.iterdir()):
        raise ValueError("Mountpoint must be empty; existing local files will not be hidden")
    credentials = credentials_for(client)
    environment = dict(os.environ, AWS_ACCESS_KEY_ID=credentials.access_key, AWS_SECRET_ACCESS_KEY=credentials.secret_key,
                       AWS_REGION=client.meta.region_name or "us-east-1", AWS_EC2_METADATA_DISABLED="true",
                       RCLONE_CONFIG_EMOVID_PROCESSED_TYPE="s3")
    if credentials.token:
        environment["AWS_SESSION_TOKEN"] = credentials.token
    else:
        environment.pop("AWS_SESSION_TOKEN", None)
    logs = Path(tempfile.gettempdir())/f"emovid-rclone-{os.getuid()}"
    logs.mkdir(exist_ok=True, mode=0o700)
    subprocess.run(["rclone", "mount", expected, str(mountpoint), "--s3-provider", "Other", "--s3-env-auth",
                    "--s3-endpoint", client.meta.endpoint_url, "--s3-region", client.meta.region_name or "us-east-1",
                    "--s3-force-path-style", "--vfs-cache-mode", "off", "--dir-cache-time", "5s",
                    "--attr-timeout", "1s", "--poll-interval", "0", "--cache-dir", str(logs/"cache"),
                    "--log-file", str(logs/"mount.log"), "--daemon", "--daemon-wait", "15s"],
                   env=environment, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.ismount(mountpoint):
        raise RuntimeError(f"Mount did not become active; inspect {logs/'mount.log'}")
    return mountpoint
