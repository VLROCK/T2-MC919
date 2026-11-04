"""Planta baixa, nuvem 3D e vídeo de navegação pelas poses COLMAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .model import load_model


def _equal_3d(ax, xyz: np.ndarray) -> None:
    if not len(xyz):
        return
    lo, hi = np.percentile(xyz, [1,99], axis=0)
    center = (lo+hi)/2
    half = max((hi-lo).max()/2, .1)
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)


def save_figures(out: Path, xyz: np.ndarray, rgb: np.ndarray,
                 cameras: np.ndarray, camera_to_world: np.ndarray,
                 evidence: list[dict], report: dict) -> None:
    units = "m" if report["scale_m_per_colmap_unit"] else "unidades SfM"
    if len(xyz) > 40000:
        rng = np.random.default_rng(919)
        indices = rng.choice(len(xyz), 40000, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]
    colors = rgb/255 if len(rgb) else np.empty((0,3))
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)
    if len(xyz):
        ax.scatter(xyz[:,0], xyz[:,1], s=.35, c=colors, alpha=.6, rasterized=True)
    ax.plot(cameras[:,0], cameras[:,1], color="#1379bc", linewidth=1.4, label="Trajetória")
    ax.scatter(cameras[[0,-1],0], cameras[[0,-1],1], c=["green","red"], s=55,
               label="Início / fim", zorder=4)
    for item in evidence:
        x,y,_ = item["xyz"]
        ax.scatter([x],[y], c="#d82727", marker="x", s=100)
        ax.annotate(item["label"], (x,y), xytext=(5,5), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.set_title("Planta projetada | conferência das paredes necessária")
    ax.grid(alpha=.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out/"floorplan.png")
    plt.close(fig)

    fig = plt.figure(figsize=(10,8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    if len(xyz):
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2], s=.35, c=colors, alpha=.6, rasterized=True)
    ax.plot(cameras[:,0],cameras[:,1],cameras[:,2], c="#1379bc", lw=2)
    ax.scatter(cameras[:,0],cameras[:,1],cameras[:,2], c="red", s=3)
    extent = np.ptp(np.vstack((xyz,cameras)),axis=0).max()
    size = max(float(extent)*.035,.05)
    corners = np.array([[-.6,-.4,1],[.6,-.4,1],[.6,.4,1],[-.6,.4,1]])*size
    for i in range(0,len(cameras),max(1,len(cameras)//25)):
        vertices = cameras[i] + (camera_to_world[i] @ corners.T).T
        for vertex in vertices:
            ax.plot([cameras[i,0],vertex[0]],[cameras[i,1],vertex[1]],
                    [cameras[i,2],vertex[2]],c="#b32323",lw=.6)
        ring = np.vstack((vertices,vertices[0]))
        ax.plot(ring[:,0],ring[:,1],ring[:,2],c="#b32323",lw=.6)
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.set_zlabel(f"Z ({units})")
    ax.set_title("Nuvem esparsa e poses da câmera")
    _equal_3d(ax, np.vstack((xyz,cameras)))
    fig.tight_layout()
    fig.savefig(out/"scene3d.png")
    plt.close(fig)

    if len(xyz) > 30:
        fig, ax = plt.subplots(figsize=(10,8),dpi=150)
        counts,xedges,yedges = np.histogram2d(xyz[:,0],xyz[:,1],bins=70)
        positive = counts[counts>0]
        cap = np.percentile(positive,98) if len(positive) else 1
        im = ax.imshow(np.clip(counts.T,0,cap),origin="lower",extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
                       cmap="magma",aspect="equal")
        ax.plot(cameras[:,0],cameras[:,1],color="cyan",lw=1)
        ax.set_title("Densidade de pontos projetados | indicador de cobertura")
        ax.set_xlabel(f"X ({units})")
        ax.set_ylabel(f"Y ({units})")
        fig.colorbar(im,ax=ax,label="Pontos por célula")
        fig.tight_layout()
        fig.savefig(out/"coverage.png")
        plt.close(fig)


def make_video(model_path: Path, metrics_path: Path, output: Path,
               seconds: int = 35, fps: int = 10) -> None:
    model = load_model(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    from .analysis import upright_rotation
    u = upright_rotation(model.images)
    scale = metrics["scale_m_per_colmap_unit"] or 1.0
    images = model.ordered_images
    centers = np.array([u @ (im.center*scale) for im in images])
    rots = Rotation.from_matrix(np.array([im.rotation @ u.T for im in images]))
    if len(images) < 2 or not len(model.points):
        raise ValueError("Vídeo exige pelo menos duas poses e pontos reconstruídos.")
    if len(model.points) > 60000:
        rng = np.random.default_rng(919)
        all_points = list(model.points.values())
        selected = [all_points[i] for i in rng.choice(len(all_points), 60000, replace=False)]
    else:
        selected = list(model.points.values())
    xyz = np.array([u @ (p.xyz*scale) for p in selected])
    bgr = np.array([p.rgb[::-1] for p in selected], dtype=np.uint8)
    slerp = Slerp(np.arange(len(images)), rots)
    nframes = seconds*fps
    width, height = 1280, 720
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV não abriu o codificador MP4; instale FFmpeg ou ajuste os codecs.")
    try:
        for frame_id in range(nframes):
            t = (len(images)-1) * frame_id/max(nframes-1,1)
            low = min(int(t),len(images)-2)
            alpha = t-low
            center = (1-alpha)*centers[low] + alpha*centers[low+1]
            rcw = slerp([t]).as_matrix()[0]
            camera_points = (rcw @ (xyz-center).T).T
            z = camera_points[:,2]
            valid = z > .08
            cp = camera_points[valid]
            colors = bgr[valid]
            camera = model.cameras[images[low].camera_id]
            f = float(camera.params[0]) * width/camera.width
            x = np.rint(f*cp[:,0]/cp[:,2] + width/2).astype(int)
            y = np.rint(f*cp[:,1]/cp[:,2] + height/2).astype(int)
            inside = (x>=1)&(x<width-1)&(y>=1)&(y<height-1)
            x,y,colors,depth = x[inside],y[inside],colors[inside],cp[:,2][inside]
            order = np.argsort(depth)[::-1]
            x,y,colors = x[order],y[order],colors[order]
            canvas = np.full((height,width,3), (18,21,25), dtype=np.uint8)
            for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1)):
                canvas[y+dy,x+dx] = colors
            cv2.rectangle(canvas, (0,0), (width,52), (13,15,18), -1)
            cv2.putText(canvas, "Reconstrucao 3D | cena simulada", (22,34),
                        cv2.FONT_HERSHEY_SIMPLEX, .82, (230,230,230), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Pose {low+1}/{len(images)}", (width-225,34),
                        cv2.FONT_HERSHEY_SIMPLEX, .68, (170,210,255), 2, cv2.LINE_AA)
            # Minimap keeps the moving camera understandable in a sparse cloud.
            map_size = 215
            xy = centers[:,:2]
            lo,hi = np.min(xy,axis=0),np.max(xy,axis=0)
            span = max(float(np.max(hi-lo)),.1)
            def pos(v):
                p = (v-lo)/span
                return (width-map_size-18+int(180*p[0]), height-map_size-10+int(180*(1-p[1])))
            cv2.rectangle(canvas,(width-map_size-30,height-map_size-30),(width-8,height-8),(13,15,18),-1)
            poly = np.array([pos(v) for v in xy], dtype=np.int32)
            cv2.polylines(canvas,[poly],False,(180,110,40),2)
            cv2.circle(canvas,pos(center[:2]),5,(50,60,240),-1)
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("work/main/sparse_txt"))
    p.add_argument("--metrics", type=Path, default=Path("work/main/analysis/metrics.json"))
    p.add_argument("--output", type=Path, default=Path("work/main/analysis/navigation.mp4"))
    p.add_argument("--seconds", type=int, default=35)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    if not 30 <= args.seconds <= 60:
        raise SystemExit("O vídeo solicitado pelo enunciado deve ter 30 a 60 segundos.")
    make_video(args.model,args.metrics,args.output,args.seconds,args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
