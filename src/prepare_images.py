"""Une pastas de fotos, preserva a ordem e seleciona vistas para SfM.

Copia somente as fotos selecionadas e registra as origens em CSV.
Marca fotos com ArUco para que a seleção não perca a referência de escala.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import re

import cv2
import numpy as np


def order_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return (int(match.group(1)) if match else 0, path.name.lower())


def choose_indices(count: int, target: int, marker: list[int]) -> list[int]:
    if target >= count:
        return list(range(count))
    mandatory = {0, count-1, *marker}
    if len(mandatory) > target:
        # Keep the complete marker segment; exceeding target is preferable to losing scale.
        return sorted(mandatory)
    uniform = np.linspace(0,count-1,target,dtype=int)
    chosen = set(mandatory)
    chosen.update(uniform.tolist())
    if len(chosen) > target:
        removable = sorted(chosen-mandatory, key=lambda i: min(abs(i-j) for j in mandatory))
        for i in removable[:len(chosen)-target]:
            chosen.remove(i)
    elif len(chosen) < target:
        for i in range(count):
            if i not in chosen:
                chosen.add(i)
                if len(chosen) == target:
                    break
    return sorted(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/images'))
    p.add_argument('--target',type=int,default=400)
    p.add_argument('--scan-limit',type=int,default=None,
                   help='Limita busca ArUco às primeiras N fotos; usar somente após confirmar que as demais não têm marcador.')
    p.add_argument('--marker-id',type=int,default=0)
    p.add_argument('--dictionary',default='DICT_4X4_50')
    args=p.parse_args()
    if args.target < 3:
        raise SystemExit('--target precisa ser >= 3')
    photos=sorted([f for source in args.sources for f in source.rglob('*')
                   if f.suffix.lower() in {'.jpg','.jpeg','.png'}],key=order_key)
    if len(photos)<3:
        raise SystemExit('Menos de 3 fotos encontradas.')
    names=[f.name.lower() for f in photos]
    if len(names)!=len(set(names)):
        raise SystemExit('Nomes repetidos entre pastas; renomeie ou use outro conjunto.')
    for dest in (args.output,):
        if dest.exists() and any(dest.iterdir()):
            raise SystemExit(f'{dest} já contém arquivos. Escolha outra saída.')
    dictionary_id=getattr(cv2.aruco,args.dictionary,None)
    if dictionary_id is None:
        raise SystemExit(f'Dicionário ArUco inválido: {args.dictionary}')
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary_id))
    records=[]
    marker_indices=[]
    for i,photo in enumerate(photos):
        if args.scan_limit is not None and i >= args.scan_limit:
            records.append({'index':i,'file':photo.name,'source':str(photo),'marker':False,
                            'sharpness_proxy':None})
            continue
        image=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f'Não foi possível ler {photo}')
        # A primeira tentativa em meia resolução é rápida; a segunda preserva marcadores pequenos.
        small=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
        _,ids,_=detector.detectMarkers(small)
        if ids is None or args.marker_id not in ids.ravel():
            _,ids,_=detector.detectMarkers(image)
        has_marker=ids is not None and args.marker_id in ids.ravel()
        if has_marker:
            marker_indices.append(i)
        thumb=cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA)
        sharpness=float(cv2.Laplacian(thumb,cv2.CV_64F).var())
        records.append({'index':i,'file':photo.name,'source':str(photo),'marker':bool(has_marker),
                        'sharpness_proxy':round(sharpness,2)})
        if (i+1)%100==0:
            print(f'Analisadas {i+1}/{len(photos)} fotos; ArUco em {len(marker_indices)}',flush=True)
    selected=set(choose_indices(len(photos),args.target,marker_indices))
    args.output.mkdir(parents=True,exist_ok=True)
    for i,photo in enumerate(photos):
        if i in selected:
            shutil.copy2(photo,args.output/photo.name)
        records[i]['selected']=i in selected
    manifest={'total_photos':len(photos),'selected_photos':len(selected),
              'marker_photos':len(marker_indices),'marker_selected':sum(i in selected for i in marker_indices),
              'sources':[str(s) for s in args.sources],
              'selected_output':str(args.output),'scan_limit':args.scan_limit}
    (args.output.parent/'image_preparation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (args.output.parent/'image_preparation.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
