"""Compile the mathematical report and figure booklet with real LaTeX engines.

TeX Live/latexmk is used for local proofs. Remote production can use a pinned
portable Tectonic binary installed only inside the campaign's results runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

CAMPAIGN=Path(__file__).resolve().parent
VERSION='0.15.0'
ASSET=f'tectonic-{VERSION}-x86_64-unknown-linux-musl.tar.gz'
URL=f'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/{ASSET}'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(runtime):
    runtime.mkdir(parents=True,exist_ok=True)
    executable=runtime/'tectonic'
    if executable.exists():return executable
    archive=runtime/ASSET
    if not archive.exists():
        for url in [URL,'https://ghfast.top/'+URL]:
            try:
                with urllib.request.urlopen(url,timeout=45) as src,archive.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                break
            except Exception:
                archive.unlink(missing_ok=True)
        else:raise RuntimeError('Unable to download the pinned Tectonic compiler')
    with tarfile.open(archive,'r:gz') as tar:
        member=next(m for m in tar.getmembers() if Path(m.name).name=='tectonic' and m.isfile())
        with tar.extractfile(member) as src,executable.open('wb') as dst:
            shutil.copyfileobj(src,dst)
    executable.chmod(0o755)
    (runtime/'compiler_download.json').write_text(json.dumps({'url':URL,'archive_sha256':digest(archive)},indent=2)+'\n')
    return executable


def compile_all(source_dir,output_dir,runtime,engine,bootstrap_remote=False):
    source_dir=source_dir.resolve();output_dir=output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True);runtime.mkdir(parents=True,exist_ok=True)
    compiler=shutil.which('latexmk') if engine=='latexmk' else shutil.which('tectonic')
    if engine=='tectonic' and not compiler and bootstrap_remote:compiler=str(bootstrap(runtime/'compiler'))
    if not compiler:raise RuntimeError(f'{engine} is unavailable')
    env=os.environ.copy();env['LC_ALL']='C';env['LANG']='C'
    if engine=='tectonic':env['TECTONIC_CACHE_DIR']=str((runtime/'cache').resolve())
    version=subprocess.run([compiler,'--version'],capture_output=True,text=True,env=env,check=True).stdout.strip()
    outputs=[]
    for name in ['report','evidence_figures']:
        if engine=='latexmk':
            command=[compiler,'-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error',
                     '-outdir='+str(output_dir),name+'.tex']
        else:
            command=[compiler,'--keep-logs','--keep-intermediates','--outdir',str(output_dir),name+'.tex']
        log=runtime/(name+'_'+engine+'.console.log')
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=source_dir,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} compilation failed; see {log}')
        pdf=output_dir/(name+'.pdf')
        if not pdf.is_file():raise RuntimeError(f'Missing output: {pdf}')
        outputs.append({'name':pdf.name,'sha256':digest(pdf),'bytes':pdf.stat().st_size})
        print(name,'compiled',pdf.stat().st_size,'bytes',flush=True)
    metadata={'engine':engine,'version':version,'outputs':outputs,
              'source_sha256':{p.name:digest(p) for p in [source_dir/'report.tex',source_dir/'evidence_figures.tex']}}
    (output_dir/'typesetting.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if output_dir==source_dir:
        for extension in ['aux','log','out','fdb_latexmk','fls','synctex.gz']:
            for name in ['report','evidence_figures']:
                p=output_dir/(name+'.'+extension)
                if p.exists():shutil.move(str(p),str(runtime/p.name))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=CAMPAIGN/'investigation')
    p.add_argument('--output',type=Path)
    p.add_argument('--runtime',type=Path,default=Path('/root/ruivi/results/ksivi_variance_investigation_20261006/runtime/typesetting'))
    p.add_argument('--engine',choices=['latexmk','tectonic'],default='latexmk')
    p.add_argument('--bootstrap-tectonic',action='store_true')
    a=p.parse_args();compile_all(a.source,a.output or a.source,a.runtime,a.engine,a.bootstrap_tectonic)


if __name__=='__main__':main()
