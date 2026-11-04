"""Acrescenta nuvem densa e textura a uma execução esparsa já concluída."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .pipeline import execute


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('work/main'))
    p.add_argument('--colmap',default='colmap')
    p.add_argument('--mesh',action='store_true')
    p.add_argument('--max-image-size',type=int,default=2000)
    args=p.parse_args()
    manifest_path=args.work/'manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'Execução esparsa ausente: {manifest_path}')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    dense=args.work/'dense'
    dense.mkdir(exist_ok=True)
    log=args.work/'commands.log'
    timing=manifest.setdefault('timing_seconds',{})
    if not (dense/'images').exists():
        timing['image_undistorter_s']=execute([
            args.colmap,'image_undistorter','--image_path',manifest['images_used'],
            '--input_path',manifest['selected_model'],'--output_path',str(dense),
            '--output_type','COLMAP','--max_image_size',str(args.max_image_size)],log)
    if not (dense/'stereo'/'depth_maps').exists() or not any((dense/'stereo'/'depth_maps').iterdir()):
        timing['patch_match_stereo_s']=execute([
            args.colmap,'patch_match_stereo','--workspace_path',str(dense),
            '--workspace_format','COLMAP','--PatchMatchStereo.geom_consistency','true'],log)
    fused=dense/'fused.ply'
    if not fused.is_file():
        timing['stereo_fusion_s']=execute([
            args.colmap,'stereo_fusion','--workspace_path',str(dense),
            '--workspace_format','COLMAP','--input_type','geometric',
            '--output_path',str(fused)],log)
    manifest['dense_ply']=str(fused)
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.mesh:
        mesh=dense/'meshed-delaunay.ply'
        if not mesh.is_file():
            timing['delaunay_mesher_s']=execute([
                args.colmap,'delaunay_mesher','--input_path',str(dense),
                '--output_path',str(mesh)],log)
        textured=dense/'textured'
        if not textured.is_dir() or not any(textured.iterdir()):
            timing['mesh_texturer_s']=execute([
                args.colmap,'mesh_texturer','--workspace_path',str(dense),
                '--input_path',str(mesh),'--output_path',str(textured)],log)
        manifest['mesh_ply']=str(mesh)
        manifest['textured_dir']=str(textured)
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'dense_ply':manifest['dense_ply'],'textured_dir':manifest.get('textured_dir')},indent=2))

if __name__=='__main__':
    main()
