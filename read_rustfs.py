"""List or download authorized RustFS objects using local S3 credentials."""

import argparse
import configparser
import getpass
import os
import socket
import sys
import tempfile
from pathlib import Path

ENDPOINT = "http://truenas-s658v2.omni-intel.cn:9000"
FALLBACK_ENDPOINT = "http://172.16.4.2:9000"
REGION = "us-east-1"
PREFIX = (
    "EMOVID_RAW/20261022161317_20261022_lwh/"
    "20261022161317_20261022_lwh/lwh/"
)


def configure(profile):
    """Store credentials outside the repository, preserving other profiles."""
    if not sys.stdin.isatty():
        raise ValueError("请在服务器的交互式终端执行 --configure，密钥不要发到聊天里。")
    access_key = getpass.getpass("RustFS Access Key（隐藏输入）: ").strip()
    secret_key = getpass.getpass("RustFS Secret Key（隐藏输入）: ").strip()
    if not access_key or not secret_key:
        raise ValueError("AK 和 SK 都不能为空。")
    credentials = Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials")).expanduser()
    credentials.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = configparser.RawConfigParser()
    if credentials.exists():
        data.read(credentials)
    if not data.has_section(profile):
        data.add_section(profile)
    data.set(profile, "aws_access_key_id", access_key)
    data.set(profile, "aws_secret_access_key", secret_key)
    # This flow uses static AK/SK, so discard any expired token in this profile.
    data.remove_option(profile, "aws_session_token")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=credentials.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            data.write(output)
        os.replace(temporary, credentials)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"凭据已保存到 {credentials}，profile={profile}，权限 600。")


def client(profile):
    import boto3
    from botocore.config import Config

    # RustFS credentials are local; do not probe AWS instance metadata.
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    available = boto3.Session().available_profiles
    if profile in available:
        session = boto3.Session(profile_name=profile)
    elif os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        session = boto3.Session()
    else:
        raise ValueError(
            f"未找到 {profile} 凭据。先运行：python read_rustfs.py --configure"
            + (f" --profile {profile}" if profile != "rustfs" else "")
        )
    endpoint = ENDPOINT
    try:
        socket.getaddrinfo("truenas-s658v2.omni-intel.cn", 9000)
    except socket.gaierror:
        endpoint = FALLBACK_ENDPOINT
    return session.client(
        "s3", endpoint_url=endpoint, region_name=REGION,
        config=Config(
            signature_version="s3v4", s3={"addressing_style": "path"},
            proxies={}, connect_timeout=5, read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configure", action="store_true", help="隐藏输入 AK/SK 并保存到本地凭据文件")
    parser.add_argument("--profile", default="rustfs")
    parser.add_argument("--bucket", default="oi-data")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--limit", type=int, default=20, help="最多列出的对象数量，默认 20")
    parser.add_argument("--key", help="要下载的完整对象 Key；不含 s3://bucket/")
    parser.add_argument("--output", type=Path, help="下载文件保存路径")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于 0")
    if args.key is not None and args.output is None:
        parser.error("下载时请同时指定 --key 和 --output")
    if args.key is None and args.output is not None:
        parser.error("--output 需要和 --key 一起使用")
    try:
        if args.configure:
            configure(args.profile)
            return 0
        s3 = client(args.profile)
        if args.key is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents overwriting an existing local file.
            with args.output.open("xb") as output:
                try:
                    s3.download_fileobj(args.bucket, args.key, output)
                except Exception:
                    args.output.unlink(missing_ok=True)
                    raise
            print(f"已下载到 {args.output}")
            return 0
        pages = s3.get_paginator("list_objects_v2").paginate(
            Bucket=args.bucket, Prefix=args.prefix,
            PaginationConfig={"MaxItems": args.limit, "PageSize": min(args.limit, 1000)},
        )
        count = 0
        for page in pages:
            for item in page.get("Contents", []):
                print(f"{item['Size']:>12} bytes  {item['Key']}")
                count += 1
        print(f"共列出 {count} 个对象（上限 {args.limit}）。")
        return 0
    except (ValueError, OSError, configparser.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        from botocore.exceptions import BotoCoreError, ClientError
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            print(f"RustFS 请求失败：{code}。请核对凭据、桶和对象权限。", file=sys.stderr)
            return 1
        if isinstance(exc, BotoCoreError):
            print(f"RustFS 连接或认证失败：{type(exc).__name__}。", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
