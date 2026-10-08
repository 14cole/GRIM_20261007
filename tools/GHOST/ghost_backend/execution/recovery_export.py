"""Join verified single-frequency exports without retaining a whole field grid."""
import base64
import json
import os
from pathlib import Path
import shutil
import uuid
import zipfile

import numpy as np
from ghost_backend.execution.recovery import RecoveryRun, sha

GRID_KEYS = {'rcs_power', 'rcs_phase', 'rcs_amp_real', 'rcs_amp_imag', 'combination_estimate_power'}
BODY_KEYS = {'body_model_amp_vv_real', 'body_model_amp_vv_imag',
             'body_model_amp_hh_real', 'body_model_amp_hh_imag'}


def _write_array(archive, name, value):
    with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def _write_text_file(archive, name, path):
    # UTF-32 matches NumPy's little-endian Unicode scalar storage. The JSON is
    # built on disk, so sample diagnostics need not form one giant Python str.
    size = path.stat().st_size
    with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
        np.lib.format.write_array_header_2_0(stream, dict(descr='<U'+str(size//4),
                                                       fortran_order=False, shape=()))
        with path.open('rb') as source:
            shutil.copyfileobj(source, stream, 1024*1024)


def _metadata_file(paths, result, path, bor):
    from ghost_backend.io.grim import _json_safe
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    def envelope(p):
        with np.load(p, allow_pickle=False) as values:
            return json.loads(str(values['solver_metadata_json']))
    first = envelope(paths[0])
    with path.open('w', encoding='utf-32-le', newline='') as stream:
        def write(value):
            for piece in encoder.iterencode(_json_safe(value)):
                stream.write(piece)
        stream.write('{')
        for index, key in enumerate(sorted(first)):
            if index: stream.write(',')
            write(key); stream.write(':')
            if key == 'metadata' and bor:
                stream.write('{"co_solved_polarizations":["VV","HH"],"per_frequency":{')
                for file_index, p in enumerate(paths):
                    for entry_index, (frequency, value) in enumerate(envelope(p)['metadata']['per_frequency'].items()):
                        if file_index or entry_index: stream.write(',')
                        write(frequency);stream.write(':');write(value)
                stream.write('},"run_recovery":');write(result['metadata']['run_recovery'])
                for extra in ('partial_result','remaining_frequencies_ghz','requested_frequency_count'):
                    if extra in result['metadata']:
                        stream.write(',');write(extra);stream.write(':');write(result['metadata'][extra])
                stream.write('}')
            elif key == 'metadata':
                write(result['metadata'])
            elif key == 'sample_diagnostics':
                stream.write('[');count=0
                for p in paths:
                    for row in envelope(p).get(key, []) or []:
                        if count:stream.write(',')
                        write(row);count+=1
                stream.write(']')
            else:
                write(first[key])
        stream.write('}')


def _inputs_file(run, path):
    with path.open('w', encoding='utf-32-le', newline='') as stream:
        stream.write('{"manifest":')
        json.dump(run.manifest, stream, sort_keys=True)
        stream.write(',"files_base64":{')
        for index, name in enumerate(run.manifest['input_files']):
            if index:stream.write(',')
            json.dump(name,stream);stream.write(':')
            json.dump(base64.b64encode((run.directory/'inputs'/name).read_bytes()).decode('ascii'),stream)
        stream.write('}}')


def join_frequency_files(paths, output, result, run, workspace):
    """Stream frequency planes into Fortran-order NPY entries inside GRIM."""
    with np.load(paths[0], allow_pickle=False) as data:
        keys = data.files
        reference = {key:data[key] for key in keys if key not in GRID_KEYS|BODY_KEYS|{'solver_metadata_json'}}
        grids = {key:(data[key].shape, data[key].dtype) for key in keys if key in GRID_KEYS|BODY_KEYS}
    frequencies = []
    for path in paths:
        with np.load(path,allow_pickle=False) as values:
            if set(values.files) != set(keys):
                raise ValueError('Frequency exports have different fields.')
            if len(values['frequencies']) != 1:
                raise ValueError('Expected one frequency in each intermediate export.')
            frequencies.append(float(values['frequencies'][0]))
            for key,value in reference.items():
                if key == 'requested_radar_grid_json':
                    declared = json.loads(str(values[key]))
                    expected = json.loads(str(value))
                    if declared.pop('frequencies_ghz') != [frequencies[-1]]:
                        raise ValueError('Radar-grid metadata disagrees with its frequency.')
                    expected.pop('frequencies_ghz')
                    if declared != expected:
                        raise ValueError('Frequency exports disagree on radar-grid settings.')
                    continue
                if key != 'frequencies' and not np.array_equal(values[key],value):
                    raise ValueError('Frequency exports disagree on '+key)
            for key,(shape,dtype) in grids.items():
                value=values[key]
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError('Frequency exports disagree on '+key+' dimensions or precision.')
    if frequencies != sorted(set(frequencies)):
        raise ValueError('Final export requires distinct, sorted frequencies.')
    if 'requested_radar_grid_json' in reference:
        declared = json.loads(str(reference['requested_radar_grid_json']))
        declared['frequencies_ghz'] = frequencies
        reference['requested_radar_grid_json'] = np.asarray(json.dumps(
            declared, sort_keys=True, separators=(',', ':')))
    metadata=workspace/'metadata.utf32'
    _metadata_file(paths,result,metadata,run.manifest['solver_kind']=='bor')
    inputs=workspace/'inputs.utf32'
    _inputs_file(run,inputs)
    with open(output,'wb') as handle:
        with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for key,value in reference.items():
                _write_array(archive,key,np.asarray(frequencies,dtype=value.dtype) if key=='frequencies' else value)
            for key,(shape,dtype) in grids.items():
                axis=2 if key in GRID_KEYS else 1
                expected_ndim=4 if key in GRID_KEYS else 2
                if len(shape)!=expected_ndim or shape[axis]!=1:
                    raise ValueError('Unsupported frequency-grid layout: '+key)
                target=list(shape);target[axis]=len(frequencies)
                with archive.open(key+'.npy','w',force_zip64=True) as stream:
                    np.lib.format.write_array_header_2_0(stream,dict(descr=np.lib.format.dtype_to_descr(dtype),
                                                                    fortran_order=True,shape=tuple(target)))
                    for polarization in range(shape[3] if axis==2 else 1):
                        for path in paths:
                            with np.load(path,allow_pickle=False) as values:
                                grid=values[key]
                                plane=grid[:,:,0,polarization] if axis==2 else grid[:,0]
                                stream.write(plane.tobytes(order='F'))
            _write_text_file(archive,'solver_metadata_json',metadata)
            _write_text_file(archive,'recovery_inputs_json',inputs)
        handle.flush();os.fsync(handle.fileno())
    # CRC validation streams each member; checking headers avoids loading the
    # combined grids back into RAM just to verify their shapes.
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise IOError('Final GRIM archive failed its integrity check.')
        for key,(shape,dtype) in grids.items():
            with archive.open(key+'.npy') as stream:
                np.lib.format.read_magic(stream)
                actual_shape,_,actual_dtype=np.lib.format.read_array_header_2_0(stream)
            target=list(shape);target[2 if key in GRID_KEYS else 1]=len(frequencies)
            if actual_shape!=tuple(target) or actual_dtype!=dtype:
                raise IOError('Final GRIM array header mismatch: '+key)


def export_run(result, output_paths, export_frequency):
    """Export this run only, transactionally, then record verified final files."""
    from ghost_backend.geometry.io import AtomicFileTransaction
    run=RecoveryRun.open(result['_recovery_run'])
    run.verify_inputs()
    frequencies=sorted(set(run.completed()))
    expected=result['metadata']['run_recovery']['completed']
    if not frequencies or expected!=sum(f in frequencies for f in run.manifest['frequencies_ghz']):
        raise IOError('Recovery outputs changed since this result was opened. Open recovery again.')
    # Final outputs must outlive automatic removal of their recovery folder.
    for output in output_paths:
        if Path(output).resolve().is_relative_to(run.directory.resolve()):
            raise ValueError('Choose a final output location outside the run recovery folder.')
    work=run.directory/('export_'+uuid.uuid4().hex)
    work.mkdir()
    transaction=AtomicFileTransaction()
    try:
        parts=[]
        for index,frequency in enumerate(frequencies):
            single=run.load(frequency)
            paths=export_frequency(single,str(work/('frequency_{:06d}.grim'.format(index))))
            if len(paths)!=len(output_paths):
                raise ValueError('Frequency exports do not share the same incidence files.')
            parts.append(paths)
            del single
        records=[]
        for index,output in enumerate(output_paths):
            joined=work/('joined_{:06d}.grim'.format(index))
            join_frequency_files([p[index] for p in parts],joined,result,run,work)
            transaction.stage_copy(joined,output)
            records.append(dict(path=str(Path(output).resolve()),sha256=sha(joined)))
        transaction.publish()
        for item in records:
            if sha(item['path'])!=item['sha256']:
                raise IOError('Published GRIM file failed verification: '+item['path'])
        transaction.commit()
        complete=set(frequencies)==set(run.manifest['frequencies_ghz'])
        run.status('exported' if complete else 'partial_exported', exports=records)
        return [item['path'] for item in records]
    except BaseException:
        transaction.abort()
        raise
    finally:
        # Only this call's generated export workspace is removed; original
        # frequency records remain until final-output verification and release.
        target=work.resolve()
        if target.parent==run.directory.resolve() and target.name.startswith('export_'):
            shutil.rmtree(target,ignore_errors=True)
