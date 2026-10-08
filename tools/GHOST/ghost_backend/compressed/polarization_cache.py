"""Finalize two polarized operators from each shared geometry tile query."""
from ghost_backend.compressed.operator import StreamedOperator, TileWriter, tile_values, take_pilot, assembly_tasks, tile_batch_values
from pathlib import Path
import numpy as np
import os,tempfile,hashlib
from ghost_backend.execution.metrics import timed_stage


class SpooledOperator(StreamedOperator):
    """Keep finalized compressed tiles on disk until this polarization is needed."""
    def __init__(self,*args,**kwargs):
        directory=Path(kwargs.pop('directory')).resolve()
        assemble=kwargs.pop('assemble',True)
        super().__init__(*args,assemble=False,**kwargs)
        fd,path=tempfile.mkstemp(prefix='ghost-tm-',suffix='.bin',dir=str(directory))
        self.path=Path(path).resolve()
        if self.path.parent!=directory:
            os.close(fd);raise RuntimeError('Unexpected spool location.')
        self.file=os.fdopen(fd,'w+b');self.records={};self.spool_bytes=0
        self.loaded=False
        if assemble:
            try:self.assemble_tiles(args[0])
            except BaseException:
                self.close();raise
    @timed_stage('compressed_spool_write')
    def store_tile(self,compressed):
        super().store_tile(compressed)
        i,j=compressed[:2]
        payload=self.tiles.pop((i,j));records=[]
        for value in payload:
            if value is None:records.append(None);continue
            raw=value.tobytes(order='C')
            records.append((value.shape,self.file.tell(),value.size,hashlib.sha256(raw).digest()))
            self.file.write(raw);self.spool_bytes+=value.nbytes
        self.records[i,j]=records;self.tiles[i,j]=None
    @timed_stage('compressed_spool_read')
    def load(self):
        if self.loaded:return
        if self.file is None:raise ValueError('Compressed spool is closed.')
        try:
            self.file.flush()
            for key,records in self.records.items():
                self.checkpoint();payload=[]
                for record in records:
                    if record is None:payload.append(None);continue
                    shape,offset,count,digest=record;self.file.seek(offset)
                    value=np.fromfile(self.file,dtype=complex,count=count)
                    if value.size!=count:raise IOError('Truncated compressed spool.')
                    if hashlib.sha256(value.tobytes()).digest()!=digest:raise IOError('Compressed spool checksum mismatch.')
                    payload.append(value.reshape(shape))
                self.tiles[key]=tuple(payload)
        except BaseException:


            self.tiles.clear();self.records.clear();self.close()
            raise
        self.records.clear();self.loaded=True;self.close()
        self.evidence['spooled_bytes']=self.spool_bytes
    def _get(self,rows,cols,row_plan=None,col_plan=None):
        if not self.loaded:raise ValueError('Load the compressed spool before querying it.')
        return super()._get(rows,cols,row_plan,col_plan)
    def block_matmul(self,rows,cols,x,row_plan=None,col_plan=None,trans=0):
        if not self.loaded:raise ValueError('Load the compressed spool before multiplying it.')
        return super().block_matmul(rows,cols,x,row_plan,col_plan,trans=trans)
    def matmul(self,b,trans=0):
        if not self.loaded:raise ValueError('Load the compressed spool before multiplying it.')
        return super().matmul(b,trans)
    def iter_tiles(self):
        if not self.loaded:raise ValueError('Load the compressed spool before reading its tiles.')
        return super().iter_tiles()
    def close(self):
        file,self.file=getattr(self,'file',None),None
        try:
            if file is not None:file.close()
        finally:
            if getattr(self,'path',None) is not None and self.path.exists():self.path.unlink()
    def __del__(self):
        try:self.close()
        except OSError:pass


def _store(operators,budget,index):
    def store(compressed):
        target=operators[index]
        target.budget=budget-operators[1-index].bytes
        target.store_tile(compressed)
    return store


def build_pair(oracle,coordinates,tile=512,budget=512*1024**2,checkpoint=None,spool_directory=None):
    operators=[]
    try:
        for index,o in enumerate(oracle.oracles):
            cls=SpooledOperator if index==1 and spool_directory is not None else StreamedOperator
            extra={'directory':spool_directory} if cls is SpooledOperator else {}
            operators.append(cls(o,coordinates,tile=tile,budget=budget,checkpoint=checkpoint,assemble=False,**extra))
            from ghost_backend.compressed.pilots import take
            operators[-1].pilot_tiles = take(getattr(o,'pilot_identity',None), operators[-1])
        from ghost_backend.compressed.tile_processes import prepare, compressed_tiles
        workers,payload=prepare(oracle,operators)
        if workers:
            for results in compressed_tiles(oracle,operators,workers,payload,checkpoint or (lambda:None)):
                for index in range(2):_store(operators,budget,index)(results[index])
        else:
            with TileWriter() as writer:
                for task in assembly_tasks(oracle,operators):
                    if hasattr(oracle,'prepare_columns'):oracle.prepare_columns(operators[0].groups[task[0][1]])
                    batches=tile_batch_values(oracle,operators[0].groups,task)
                    for (i,j,_),values in zip(task,batches):
                        known = [take_pilot(op,i,j) for op in operators]
                        for index in range(2):
                            if known[index] is not None:
                                writer.submit_prepared(known[index],_store(operators,budget,index))
                            else:
                                writer.submit(operators[index].compress_tile,_store(operators,budget,index),i,j,*values[index])
                    batches=None
        for op,source in zip(operators,oracle.oracles):op.finalize(source)
    except BaseException:
        for op in operators:
            if isinstance(op,SpooledOperator):op.close()
        raise
    return operators
