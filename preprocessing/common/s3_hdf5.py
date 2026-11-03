"""Read HDF5 directly over bounded, snapshot-pinned S3 Range requests."""

from collections import OrderedDict
from contextlib import contextmanager
import io
import operator

import h5py

from preprocessing.common.emovid_segments import split_s3


class S3RangeFile(io.RawIOBase):
    """Seekable read-only object; cache at most eight 1 MiB blocks in memory."""

    def __init__(self, client, uri, *, etag=None, block_size=1024**2, cache_blocks=8):
        super().__init__()
        if block_size <= 0 or cache_blocks <= 0:
            raise ValueError("Range block size and cache size must be positive")
        self.client = client
        self.bucket, self.key = split_s3(uri)
        head = client.head_object(Bucket=self.bucket, Key=self.key)
        self.etag = head['ETag']
        if etag is not None and self.etag != etag:
            raise ValueError("S3 object changed since its manifest was written")
        self.size = head['ContentLength']
        self.version_id = head.get('VersionId')
        self.block_size, self.cache_blocks = block_size, cache_blocks
        self.position = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        offset = operator.index(offset)
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def _block(self, number):
        if number not in self.cache:
            start = number*self.block_size
            stop = min(self.size, start+self.block_size)-1
            kwargs = dict(Bucket=self.bucket, Key=self.key,
                          Range=f'bytes={start}-{stop}', IfMatch=self.etag)
            if self.version_id:
                kwargs['VersionId'] = self.version_id
            response = self.client.get_object(**kwargs)
            try:
                value = response['Body'].read()
            finally:
                response['Body'].close()
            if len(value) != stop-start+1:
                raise OSError("Truncated S3 Range response")
            self.cache[number] = value
            if len(self.cache) > self.cache_blocks:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        return self.cache[number]

    def read(self, size=-1):
        self._checkClosed()
        stop = self.size if size is None or size < 0 else min(self.size, self.position+size)
        chunks = []
        while self.position < stop:
            number, offset = divmod(self.position, self.block_size)
            block = self._block(number)
            length = min(stop-self.position, len(block)-offset)
            chunks.append(block[offset:offset+length])
            self.position += length
        return b''.join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        self.cache.clear()
        super().close()


@contextmanager
def open_s3_hdf5(client, uri, *, etag=None):
    """Close h5py before its file object, including on exceptions."""
    with S3RangeFile(client, uri, etag=etag) as stream:
        with h5py.File(stream, 'r') as dataset:
            yield dataset
