"""Bounded disk ownership of completed fine operators between polynomial solves."""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from ghost_backend.execution.metrics import timed_stage


class RetainedOperatorSpool:
    """Own an operator's tiles on disk, retaining its original metadata/identity.

    Payload is removed only after every tile was successfully written. A failed
    restore discards partial arrays, closes the file, and lets the caller rebuild
    independently. No general-purpose pickle or second full payload is created.
    """
    bytes=0

    @timed_stage('compressed_polynomial_spool_write')
    def __init__(self,operator,directory):
        self.operator=operator;self.file=None;self.path=None;self.records=[];self.disk_bytes=0
        directory=Path(directory).resolve()
        fd,path=tempfile.mkstemp(prefix='ghost-polynomial-',suffix='.bin',dir=str(directory))
        self.path=Path(path).resolve()
        if self.path.parent!=directory:
            os.close(fd);raise RuntimeError('Unexpected polynomial spool location.')
        self.file=os.fdopen(fd,'w+b')
        try:
            for key,payload in operator.tiles.items():
                operator.checkpoint();records=[]
                for value in payload:
                    if value is None:records.append(None);continue
                    raw=value.tobytes(order='C')
                    records.append((value.shape,self.file.tell(),value.size,hashlib.sha256(raw).digest()))
                    self.file.write(raw);self.disk_bytes+=value.nbytes
                self.records.append((key,records))
            self.file.flush()
        except BaseException:
            self.close();raise
        operator.tiles.clear()
        operator.evidence['polynomial_spooled_bytes']=self.disk_bytes

    @timed_stage('compressed_polynomial_spool_read')
    def restore(self):
        operator=self.operator
        if operator is None or self.file is None:raise OSError('Retained polynomial spool is closed.')
        try:
            for key,records in self.records:
                operator.checkpoint();payload=[]
                for record in records:
                    if record is None:payload.append(None);continue
                    shape,offset,count,digest=record;self.file.seek(offset)
                    value=np.fromfile(self.file,dtype=complex,count=count)
                    if value.size!=count:raise OSError('Truncated retained polynomial spool.')
                    if hashlib.sha256(value.tobytes()).digest()!=digest:
                        raise OSError('Retained polynomial spool checksum mismatch.')
                    payload.append(value.reshape(shape))
                operator.tiles[key]=tuple(payload)
        except BaseException:
            operator.tiles.clear();self.close();raise
        self.operator=None;self.close()
        return operator

    def close(self):
        file,self.file=self.file,None
        try:
            if file is not None:file.close()
        finally:
            if self.path is not None:
                try:self.path.unlink()
                except FileNotFoundError:pass
        self.records.clear()

    def __del__(self):
        try:self.close()
        except (AttributeError,OSError):pass
