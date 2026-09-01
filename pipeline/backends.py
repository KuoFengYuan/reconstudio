"""Training-backend registry + environment resolution + preflight checks.

The panel itself stays torch-free: it never imports a trainer, it only spawns
one as a subprocess inside that trainer's own conda env. Each backend is *data*
(which env, which repo, how to build the command), so enabling a trainer on a
new machine is a config entry — not a code change. This is what makes the panel
portable: deploying it is `pip install -r requirements.txt`; the heavy CUDA env
is a per-machine prerequisite that backends.py merely *locates* and *checks*.

Interpreter resolution order for a backend (most portable first):
  1. an explicit absolute "python" in the backend spec
  2. $CONDA_ROOT/envs/<env>/bin/python
  3. <envs>/<env>/bin/python  derived from this process's own sys.prefix
  4. `conda info --base`/envs/<env>/bin/python, then common install locations

Per-machine overrides live in backends.json (gitignored; see
backends.example.json), shallow-merged over the BUILTIN_BACKENDS below.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .config import settings
from .preflight import colmap_check, make_check, probe, system_report

BASE = Path(__file__).resolve().parent.parent          # reconstudio/
COLMAP_BIN = settings.colmap_bin
BACKENDS_FILE = settings.backends_file

# Built-in specs. Zero-config when the conda env name / sibling repo / binary path
# match the defaults below; override/extend via backends.json (only the keys that
# differ on THIS machine — shallow merge is per top-level backend key, so e.g.
# overriding "exec" for lichtfeld-mrnf does NOT require re-listing its "params").
# Repos/paths are relative to BASE unless absolute.
BUILTIN_BACKENDS: dict[str, dict] = {
    "gs2m": {
        "label": "GS-2M（材質感知 → mesh）",
        "conda_env": "gs2m",
        "repo": "../GS-2M",
        "train_script": "train.py",
        # {scene}/{out} filled by train.py; {args} = the tunable params below;
        # {extra} = a free escape-hatch text field for anything not exposed.
        "train_args": "-s {scene} -m {out} {args} {extra}",
        "default_iterations": 30000,
        # GS-2M reads sparse/0 + images/; we adapt a COLMAP workspace to that shape.
        "scene_layout": "colmap_sparse0",
        # imported in the trainer env by /doctor to catch arch/CUDA mismatch early.
        "probe_imports": ["torch", "diff_gaussian_rasterization"],
        # post-training mesh extraction the user can run next (shown on "done").
        "mesh_cmd": "python render.py -m {out} --extract_mesh --skip_test",
        # Tunable params rendered as form fields — NOT hardcoded in the panel, and
        # overridable per machine via backends.json. type: int|float|str|bool|select.
        "params": [
            {"key": "iterations", "flag": "--iterations", "type": "int",
             "default": "30000", "label": "迭代次數",
             "hint": "總訓練步數。預設 30000 通常足夠;資料量大或要更精細可加到 40000–50000，但時間正比拉長。"},
            {"key": "resolution", "flag": "-r", "type": "int", "default": "1",
             "label": "解析度降採樣 (-r)",
             "hint": "1 = 用原解析度訓練（預設、品質最佳）。當影像長/寬 > 1600px，建議設 2（半解析度）以加速並省顯存;3、4 再更小。"},
            {"key": "data_device", "flag": "--data_device", "type": "select",
             "options": ["cpu", "cuda"], "default": "cpu", "label": "影像快取裝置",
             "hint": "訓練影像快取在哪。cuda 最快但吃顯存，影像多（數百張以上）容易爆顯存;放 cpu 改用系統記憶體較穩，速度差距不大。"},
            {"key": "sh_degree", "flag": "--sh_degree", "type": "int", "default": "3",
             "label": "球諧階數 (SH degree)",
             "hint": "顏色隨視角變化的能力。3 = 最高品質（預設）;調低（1–2）可省記憶體、訓練略快，但反光/視角相依的外觀會變差。"},
            {"key": "reflection_threshold", "flag": "--reflection_threshold",
             "type": "float", "default": "1.0", "label": "反射閾值 (reflection_threshold)",
             "hint": "判定「平滑/反光面」的靈敏度，影響多視角光度變化的解讀。漫反射（霧面）場景設 ≥ 1.0;反光/光澤面設 < 1.0（越低越敏感）。"},
            {"key": "lambda_normal", "flag": "--lambda_normal", "type": "float",
             "default": "0.1", "label": "法線正則權重 λ_normal",
             "hint": "影響表面完整度。重建出的 mesh 不封閉、有破洞時調高（例如 0.2–0.5）來補洞;太高則細節會被抹平。"},
            {"key": "eval", "flag": "--eval", "type": "bool", "default": False,
             "label": "切 train/test 評估 (--eval)",
             "hint": "每 8 張抽 1 張當測試集以計算 PSNR 等指標;純做 mesh 重建時可不開，讓全部影像都參與訓練。"},
            # --- group: 材質分解（只在 --material 開啟時生效） ---
            {"key": "material", "flag": "--material", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "材質分解 (--material)",
             "hint": "開啟 PBR 材質分解（albedo 反照率 / roughness 粗糙度 / 金屬度），自 iter 5000 起自動開始（不必加迭代數）。"
                     "建議開啟的時機：① 物體有反光或光澤面 — 把高光從幾何裡分離，避免反光被誤烤成凸起的假幾何;"
                     "② 你需要 relighting（重打光）或要匯出材質貼圖。純霧面物體、或只想要 mesh，就不用開，可省時間與顯存。"},
            {"key": "metallic", "flag": "--metallic", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "金屬度通道 (--metallic)",
             "hint": "僅在 --material 開啟時有效。重建金屬材質（金屬器物、金屬漆）時才建議開，多估一個金屬度通道;非金屬物體請留關。"},
            {"key": "gamma", "flag": "--gamma", "type": "bool", "default": False,
             "group": "材質分解（進階）", "label": "Gamma 色調映射 (--gamma)",
             "hint": "僅在 --material 開啟時有效。輸入影像偏暗或動態範圍大、材質分解結果過曝/過暗時可試開。"},
            {"key": "lambda_smooth", "flag": "--lambda_smooth", "type": "float",
             "default": "0.0", "group": "材質分解（進階）", "label": "粗糙度平滑 λ_smooth",
             "hint": "僅在 --material 有效。當反射線索不足、粗糙度圖雜訊大時調高，把正確判定的粗糙度往周圍傳播;預設 0 = 不啟用。"},
            # --- group: 前景遮罩（物件去背重建） ---
            {"key": "masks", "flag": "--masks", "type": "str", "default": "",
             "group": "前景遮罩（物件去背）", "label": "前景遮罩目錄 (--masks)", "placeholder": "masks 或絕對路徑",
             "hint": "物件去背重建用。填放遮罩 PNG 的資料夾名（相對 source，例如 masks）或絕對路徑;可先用 scripts/mask.py 產生。留空 = 不使用。"},
            {"key": "mask_gt", "flag": "--mask_gt", "type": "bool", "default": False,
             "group": "前景遮罩（物件去背）", "label": "遮罩 GT 影像 (--mask_gt)",
             "hint": "搭配 --masks 使用。把背景從 GT 影像挖掉、只擬合前景物件;背景雜亂、要乾淨單一物件 mesh 時很有用。"},
        ],
        # --- mesh extraction (render.py --extract_mesh) ---
        "mesh_script": "render.py",
        "mesh_args": "-m {out} --extract_mesh --skip_test {args} {extra}",
        # Optional marker-board scaling: when the Mesh form's「提供 marker」is on,
        # run_mesh runs the panel's tools/estimate_marker_scale.py + scale_mesh.py
        # (panel-owned) using this backend's env. A backend opts in just by
        # declaring the physical ChArUco board geometry below — applied
        # automatically (no manual entry). Override per machine in backends.json.
        "marker_defaults": {
            "squares_x": 9, "squares_y": 6,
            "square_mm": 28.806, "marker_mm": 21.12,
            "dict": "DICT_5X5_100",
        },
        "mesh_params": [
            {"key": "mesh_only", "flag": "--mesh_only", "type": "bool", "default": True,
             "label": "只抽 mesh（加速）",
             "hint": "跳過逐視角 PNG（render/normal/depth/PBR）輸出，只做 TSDF 融合抽 mesh，大幅加速。只想要 mesh 時建議開。"},
            {"key": "auto_voxel", "flag": "--auto_voxel", "type": "bool", "default": True,
             "label": "自動體素大小（推薦）",
             "hint": "從高斯點雲密度自動估 voxel_size（作者推薦,跨尺度最穩）。需要 scipy;不可用時自動 fallback 到啟發式。勾選時下方 voxel_size / sdf_trunc 會停用,由程式自行決定。"},
            {"key": "voxel_size", "flag": "--voxel_size", "type": "float", "default": "0.006",
             "label": "體素大小 voxel_size",
             "hint": "TSDF 取樣格大小（場景座標單位,非公分）。預設 0.006,取消上方「自動」才生效。越小 → 越密越細,但越慢、越吃記憶體。注意:此值是相對場景尺度(≈ cameras_extent/1250),換差很多的拍攝距離時要重估。"},
            {"key": "sdf_trunc", "flag": "--sdf_trunc", "type": "float", "default": "",
             "label": "SDF 截斷 sdf_trunc",
             "hint": "越小 → 表面越銳利但對深度雜訊越敏感;越大 → 越穩但細節變糊。留空 = 自動（4 × voxel_size,例如 voxel 0.006 → 0.024）。"},
            {"key": "max_depth", "flag": "--max_depth", "type": "float", "default": "",
             "label": "最大融合深度 max_depth",
             "hint": "depth fusion 時超過此距離的點丟棄。留空 = 自動（2 × 場景半徑）;背景雜訊多可調小。"},
            {"key": "num_clusters", "flag": "--num_clusters", "type": "int", "default": "1",
             "label": "保留聚類數 num_clusters",
             "hint": "後處理保留幾個連通塊。1 = 只取最大物件（預設）;場景含多個分離物件時再調高。"},
            {"key": "filter_depth", "flag": "--filter_depth", "type": "bool", "default": False,
             "label": "融合前過濾深度",
             "hint": "TSDF 融合前先過濾深度圖，可去除部分雜訊浮點。"},
        ],
    },
    "gsplat": {
        "label": "gsplat (nerfstudio)",
        "conda_env": "gsplat",
        "repo": "../gsplat",
        "train_script": "examples/simple_trainer.py",
        # 固定用 "default" 子命令 = DefaultStrategy（非 MCMC）。不要改成 "mcmc"。
        # --save_ply / --random_bkgd / --antialiased 直接寫死開啟（不在 UI 上顯示可調）。
        "train_args": "default --data_dir {scene} --result_dir {out} --save_ply --random_bkgd --antialiased {args} {extra}",
        "probe_imports": ["torch", "gsplat"],
        "params": [
            {"key": "max_steps", "flag": "--max_steps", "type": "int", "default": "30000",
             "label": "迭代次數 (max_steps)",
             "hint": "總訓練步數。預設 30000 通常足夠；資料量大或要更精細可加到 40000–50000，時間也等比拉長。"},
            {"key": "steps_scaler", "flag": "--steps_scaler", "type": "float", "default": "1",
             "label": "步數縮放 (steps_scaler)",
             "hint": "把訓練步數整體乘以此倍率，會同步縮放：迭代次數、eval/save/ply 存檔步數、SH 升階間隔，以及下方「密化策略」的 refine_start_iter / refine_stop_iter / reset_every / refine_every——不用手動一個個改。例如想跑 1/4 的步數做快速預覽，設 0.25 即可；預設 1（不縮放）。"},
            {"key": "data_factor", "flag": "--data_factor", "type": "int", "default": "1",
             "label": "影像降採樣 (data_factor)",
             "hint": "訓練影像縮小倍率。預設 1 = 用原生解析度訓練（品質最佳，顯存/時間開銷也最大）。官方 benchmark 對室內小場景（bonsai/counter/kitchen/room）用 2，室外大場景（garden/bicycle 等）用 4，速度優先時可調高。"},
            {"key": "sh_degree", "flag": "--sh_degree", "type": "int", "default": "3",
             "label": "球諧階數 (sh_degree)",
             "hint": "顏色隨視角變化的能力。3 = 最高品質（預設）。調低（0–2）可省記憶體、訓練略快，但反光/視角相依外觀會變差。"},
            {"key": "sh_fp16", "flag": "--sh_fp16", "type": "bool", "default": False,
             "label": "SH 係數用 fp16 (sh_fp16)",
             "hint": "v1.6.0 新增。球諧係數計算時轉成 fp16，可省顯存、加速，代價是精度略降。高斯數量很大、顯存吃緊時可開；一般場景不需要。"},
            {"key": "camera_model", "flag": "--camera_model", "type": "select",
             "options": ["pinhole", "fisheye", "ortho", "ftheta"], "default": "pinhole",
             "label": "相機鏡頭模型 (camera_model)",
             "hint": "v1.6.0 擴充了感測器支援。pinhole = 一般針孔相機（預設，絕大多數手機/相機直出影像）；fisheye = 魚眼鏡頭；ortho = 正交投影（空拍測繪常見）；ftheta = 特定工業寬視角鏡頭。COLMAP 影像多半維持 pinhole 即可，除非確定拍攝鏡頭是其他類型。"},
            {"key": "packed", "flag": "--packed", "type": "bool", "default": False,
             "label": "省顯存模式 (packed)",
             "hint": "用打包格式做光柵化，顯存用量較低但速度略慢。高斯數量很大、快爆顯存時可開。"},
            {"key": "sparse_grad", "flag": "--sparse_grad", "type": "bool", "default": False,
             "label": "稀疏梯度 / 稀疏光柵化 (sparse_grad)",
             "hint": "v1.6.0 強化了 active-tile 稀疏 3DGS 光柵化的反向傳播穩定度。開啟後只對可見高斯計算梯度，大場景、高斯數很多時可省顯存並加速；仍屬實驗性選項，遇到訓練不穩定可先關閉排查。"},
            # --- group: 相機 / 外觀優化 ---
            {"key": "pose_opt", "flag": "--pose_opt", "type": "bool", "default": False,
             "group": "相機 / 外觀優化", "label": "相機姿態優化 (pose_opt)",
             "hint": "訓練中微調 COLMAP 給的相機外參。COLMAP 姿態不夠準（例如影像抖動、手機隨手拍）時開，可改善銳利度；姿態已經很準時開了幫助不大甚至可能變差。"},
            {"key": "app_opt", "flag": "--app_opt", "type": "bool", "default": False,
             "group": "相機 / 外觀優化", "label": "外觀嵌入優化 (app_opt)",
             "hint": "為每張影像學一個外觀嵌入，吸收曝光/白平衡/光照隨拍攝時間變化的差異。同一場景在不同時間、不同曝光設定下拍攝（例如戶外晴陰交錯）時建議開。"},
            {"key": "post_processing", "flag": "--post_processing", "type": "select",
             "options": ["none", "bilateral_grid", "ppisp"], "default": "none",
             "group": "相機 / 外觀優化", "label": "曝光後處理 (post_processing)",
             "hint": "v1.6.0 起 PPISP 兩種後處理都能搭配目前用的 DefaultStrategy（非 MCMC）。用途和外觀嵌入優化類似，但是對「渲染結果」做各視角曝光/白平衡補償：bilateral_grid = 雙邊網格；ppisp = 感知式後處理（v1.6.0 新增，通常效果更好）。none = 不使用（預設）。發現不同視角明暗、色調不一致時可試 ppisp。"},
            # --- group: 密化策略進階（AbsGS） ---
            {"key": "strategy.absgrad", "flag": "--strategy.absgrad", "type": "bool", "default": False,
             "group": "密化策略進階（AbsGS）", "label": "絕對梯度密化 (absgrad, AbsGS)",
             "hint": "用絕對梯度取代平均梯度判斷密化，細節通常更好（AbsGS 論文）。開啟時建議把下方「密化梯度門檻」調高（例如 0.0008），否則會長出過多高斯。"},
            {"key": "strategy.grow_grad2d", "flag": "--strategy.grow-grad2d", "type": "float",
             "default": "0.0002", "group": "密化策略進階（AbsGS）", "label": "密化梯度門檻 (grow_grad2d)",
             "hint": "2D 投影梯度超過此值就 densify。預設 0.0002（搭配平均梯度）；開啟上方「絕對梯度密化」時建議調高到約 0.0008。調低 → 更容易新增高斯（細節更多但更慢、更吃顯存）。"},
            # --- group: 深度監督 ---
            {"key": "depth_loss", "flag": "--depth_loss", "type": "bool", "default": False,
             "group": "深度監督（需另有深度圖）", "label": "深度損失 (depth_loss)",
             "hint": "需搭配 COLMAP 稀疏深度或外部深度圖做監督。可加速收斂、改善低紋理區域的幾何；沒有深度資料時不要開。"},
            {"key": "depth_lambda", "flag": "--depth_lambda", "type": "float", "default": "0.01",
             "group": "深度監督（需另有深度圖）", "label": "深度損失權重 (depth_lambda)",
             "hint": "僅在「深度損失」開啟時生效。"},
        ],
    },
    # LichtFeld Studio (C++/CUDA binary, no conda env) — see README「安裝」第 3 節
    # for the build steps. Zero-config when the sibling repo sits at ../LichtFeld-Studio
    # and is built in Release; override only "exec" in backends.json if it lives
    # elsewhere on a given machine. No mesh_args — it can't extract a mesh.
    "lichtfeld-mrnf": {
        "label": "LichtFeld · MR-NF (無 mesh)",
        "launch": "binary",
        "exec": "../LichtFeld-Studio/build/LichtFeld-Studio",
        "train_args": "-d {scene} -o {out} --strategy mrnf --headless --no-splash {args} {extra}",
        "params": [
            {"key": "iterations", "flag": "--iter", "type": "int", "default": "30000", "label": "迭代次數"},
            {"key": "steps_scaler", "flag": "--steps-scaler", "type": "float", "default": "1",
             "label": "步數縮放 (預設 1;留空=用內建預設)",
             "hint": "把所有訓練步數(迭代次數 + refine/lr 排程)整體乘以此倍率。有效迭代 = 迭代次數 × 此值"
                     "(LichtFeld 內部換算,不必自己乘)。預設 1(不縮放);留空 = 用 LichtFeld 內建預設。"},
            {"key": "sh_degree", "flag": "--sh-degree", "type": "select", "options": ["0", "1", "2", "3"],
             "default": "3", "label": "SH 階數"},
            {"key": "max_cap", "flag": "--max-cap", "type": "int", "default": "1000000", "label": "最大高斯數 (max-cap)"},
            {"key": "resize_factor", "flag": "--resize_factor", "type": "select",
             "options": ["auto", "1", "2", "4", "8"], "default": "auto", "label": "影像降採樣"},
            {"key": "max_width", "flag": "--max-width", "type": "int", "default": "3840",
             "label": "影像寬度上限 px (0 = 不設限)",
             "hint": "超過這個寬度就縮到這個寬度,和「影像降採樣」是兩道獨立的關卡。"
                     "LichtFeld 內建預設就是 3840,而且不會警告 —— 航拍原圖 14204 px 會被砍成 3840,"
                     "6 cm 的 GSD 變成 23 cm。要用原生解析度訓練一定要填 0。"},
            {"key": "centralize", "flag": "--centralize", "type": "select",
             "options": ["by_cameras", "by_pointcloud", "off"], "default": "by_cameras",
             "label": "座標原點置中 (centralize)",
             "hint": "把相機與點雲整體平移到幾何中位數,並記錄位移量(world = local + world_origin)。"
                     "**大地座標(TWD97/UTM)的資料一定要開**:高斯位置是 float32,TWD97 northing "
                     "~2,652,686 的 float32 間距是 0.25 m —— 是 6 cm GSD 的 4 倍,不置中的話每個高斯"
                     "位置都被量化到 25 cm,連相對量測都會吃到這個誤差。"
                     "by_cameras=用相機位置中位數(預設;相機一定存在,航拍飛行網格無離群值最穩);"
                     "by_pointcloud=用稀疏點雲中位數(點雲是空的時候會靜默不置中,較不保險);"
                     "off=不平移。注意:置中後匯出的 PLY 是本地座標,位移量只寫進 .licht 專案檔 —— "
                     "距離/尺寸等相對量測不受影響,但要換回絕對 TWD97 座標時要把 world_origin 加回去。"},
            {"key": "use_16bit", "group": "影像與資料集", "flag": "--use-16bit", "type": "bool", "default": False, "label": "16-bit 色彩訓練",
             "hint": "只在來源本身就是 16-bit(RAW 轉出的 TIFF/PNG、HDR 素材)才有用;一般手機/相機直出的 8-bit "
                     "JPEG/PNG 開這個沒意義。自動用無損 JPEG2000 做磁碟快取,檔案較大、稍慢。預設關。"},
            {"key": "cpu_cache", "group": "影像與資料集", "flag": "--no-cpu-cache", "type": "bool", "invert": True, "default": True,
             "label": "CPU 快取", "hint": "影像快取在 CPU 記憶體(預設開)。取消勾選 = 傳 --no-cpu-cache 停用。"},
            {"key": "mask_mode", "group": "影像與資料集", "flag": "--mask-mode", "type": "select",
             "options": ["none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"], "default": "none",
             "label": "遮罩模式 (前景去背訓練)", "hint": "none=不用遮罩(一般重建)。其餘模式需另外提供遮罩影像。"},
            {"key": "bilateral_grid", "group": "外觀 / 曝光校正", "flag": "--bilateral-grid", "type": "bool", "default": False,
             "label": "雙邊網格 (曝光補償)"},
            {"key": "ppisp", "group": "外觀 / 曝光校正", "flag": "--ppisp", "type": "bool", "default": True,
             "label": "PPISP",
             "hint": "每張相機各自的物理合理 ISP 模型(曝光 + 白平衡),比雙邊網格更貼近真實相機的曝光/白平衡差異,"
                     "可與雙邊網格同時開。搭配下方「EXIF Exposure」用照片 EXIF 加速收斂。(面板預設開)"},
            {"key": "ppisp_exif_exposure", "group": "外觀 / 曝光校正", "flag": "--no-ppisp-exif-exposure", "type": "bool", "invert": True,
             "default": True, "label": "EXIF Exposure",
             "hint": "PPISP 開啟時,從照片 EXIF 讀取每張相機的曝光初始值以加速收斂(LichtFeld 預設開)。"
                     "取消勾選 = 傳 --no-ppisp-exif-exposure 停用,PPISP 曝光從零開始學。僅在勾選 PPISP 時有意義。"},
            {"key": "background_improvements", "flag": "--background-improvements", "type": "bool",
             "default": True, "label": "BG Improvements",
             "hint": "改善遠景/背景高斯的生長 —— far-field 種子與分裂、衰減緩解、"
                     "生長上限、逐高斯位置步進、依可見度比例排序生長、分批填充容量上限。(面板預設開)"},
            {"key": "enable_mip", "group": "外觀 / 曝光校正", "flag": "--enable-mip", "type": "bool", "default": True, "label": "Mip 濾波 (抗鋸齒)"},
            {"key": "use_depth_loss", "group": "深度監督", "flag": "--use-depth-loss", "type": "bool", "default": False,
             "label": "深度損失 (use-depth-loss)",
             "hint": "讀取資料集 depth/ 深度圖做監督(需與 images/ 同層,可用「🌊 深度」工具產生)。預設關。"},
            {"key": "depth_loss_mode", "group": "深度監督", "flag": "--depth-loss-mode", "type": "select",
             "options": ["ssi", "ssi-disparity", "ssi-depth"], "default": "ssi", "label": "深度損失模式",
             "hint": "深度先驗慣例:ssi=自動偵測(預設,建議);ssi-disparity/ssi-depth=強制指定先驗是反深度或正深度。"
                     "僅在勾選深度損失時生效。"},
            {"key": "depth_loss_weight", "group": "深度監督", "flag": "--depth-loss-weight", "type": "float", "default": "2.0",
             "label": "深度損失權重", "hint": "深度監督權重(LichtFeld 預設 2.0)。僅在勾選深度損失時生效。"},
            {"key": "use_normal_loss", "group": "法向量監督", "flag": "--use-normal-loss", "type": "bool", "default": False,
             "label": "法向量損失 (use-normal-loss)",
             "hint": "讀取資料集 normal/ 或 normals/ 法向量圖做監督(需與 images/ 同層、相同檔名,"
                     "PNG/TIFF,RGB 編碼 [-1,1] 法向量)。缺圖或尺寸不符時 LichtFeld 會自動用內建 "
                     "MoGe-2 即時補生成(見下方「法向量自動生成」);仍可用「🌊 深度」工具"
                     "先預生成以重複利用或用原始解析度跑一次。預設關。"},
            {"key": "normal_auto_generate", "group": "法向量監督", "flag": "--no-normal-auto-generate", "type": "bool", "invert": True,
             "default": True, "label": "法向量自動生成",
             "hint": "訓練開始前掃 normals/,缺圖或尺寸和當前訓練解析度不符的,直接用內建 MoGe-2 從全解析度 "
                     "images/ 現場補生成(LichtFeld 預設開),所以換降採樣倍率也不會踩到尺寸不符。"
                     "取消勾選 = 傳 --no-normal-auto-generate:只吃你自己預先產生的圖,缺圖就報錯 —— "
                     "想確保用的是特定一批法向量圖(例如自己跑過後處理的)時才關。僅在勾選法向量損失時有意義。"},
            {"key": "normal_loss_weight", "group": "法向量監督", "flag": "--normal-loss-weight", "type": "float", "default": "0.005",
             "label": "法向量先驗權重",
             "hint": "渲染法向量對法向量先驗的 cosine 監督權重。僅在勾選法向量損失時生效。"
                     "0.005 是上游「以畫面品質為主、讓幾何自由發展」的預設;"
                     "**若幾何本身就是產品(要抽 mesh / TSDF 融合),上游建議拉到 ~0.1**(20 倍)。"},
            {"key": "normal_consistency_weight", "group": "法向量監督", "flag": "--normal-consistency-weight", "type": "float",
             "default": "0.001", "label": "深度-法向量一致性權重",
             "hint": "渲染法向量與渲染深度反推法向量的一致性權重。僅在勾選法向量損失時生效。"},
            {"key": "normal_flatten_weight", "group": "法向量監督", "flag": "--normal-flatten-weight", "type": "float",
             "default": "0", "label": "扁平化權重",
             "hint": "法向量監督期間,讓高斯最短軸 scale 攤平的權重。僅在勾選法向量損失時生效。"},
            {"key": "normal_loss_space", "group": "法向量監督", "flag": "--normal-loss-space", "type": "select",
             "options": ["auto", "camera-opencv", "camera-opengl", "world"], "default": "auto",
             "label": "法向量先驗座標系",
             "hint": "auto=自動偵測(建議);其餘為強制指定先驗座標慣例。僅在勾選法向量損失時生效。"},
        ],
    },
}


def build_cli(params: list, values: dict) -> str:
    """Assemble CLI tokens from a param schema (list of {key,flag,type,...}) +
    submitted `values` (keyed by each param's `key`). Bools emit just the flag
    when truthy; others emit `flag value` when non-empty. Values are shell-quoted
    so paths with spaces survive the round-trip through shlex.split in train.py.

    A bool param may set `"invert": true` for "feature on by default, flag DISABLES
    it" CLIs (e.g. LichtFeld's --no-cpu-cache): the checkbox reads as the feature
    being ON (checked by default) and the flag is emitted only when UNchecked.
    Reused for both training (`params`) and mesh extraction (`mesh_params`)."""
    import shlex
    toks: list[str] = []
    for pr in params:
        v = values.get(pr["key"])
        if pr.get("type") == "bool":
            on = bool(v)
            if pr.get("invert"):
                on = not on          # checked = feature on -> don't emit the disable flag
            if on:
                toks.append(pr["flag"])
        else:
            s = ("" if v is None else str(v)).strip()
            if s:
                toks += [pr["flag"], s]
    return " ".join(shlex.quote(t) for t in toks)


# --------------------------------------------------------------------------- #
# Environment / path resolution
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def conda_envs_dir() -> Path | None:
    """Directory that holds conda environments (.../envs), or None if unknown."""
    root = settings.conda_root
    if root and (Path(root) / "envs").is_dir():
        return Path(root) / "envs"
    # We run inside <root>/envs/<panel-env>, so our sys.prefix's parent is envs/.
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs" and prefix.parent.is_dir():
        return prefix.parent
    base = probe(["conda", "info", "--base"]).strip()
    if base and (Path(base) / "envs").is_dir():
        return Path(base) / "envs"
    # Last resort: the usual install locations. Keep this list in sync with
    # detect_conda_root() in tools/detect.sh — that's the bash twin run.sh needs
    # before any Python exists, so the two can only be kept aligned by hand.
    for d in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        if (Path.home() / d / "envs").is_dir():
            return Path.home() / d / "envs"
    if Path("/opt/conda/envs").is_dir():
        return Path("/opt/conda/envs")
    return None


def env_python(spec: dict) -> Path | None:
    """Absolute python for a backend's conda env, or None if not found."""
    if spec.get("python"):
        py = Path(spec["python"])
        return py if py.is_file() else None
    envs = conda_envs_dir()
    if not envs:
        return None
    py = envs / spec["conda_env"] / "bin" / "python"
    return py if py.is_file() else None


def repo_path(spec: dict) -> Path:
    r = Path(spec.get("repo", "."))
    return r if r.is_absolute() else (BASE / r).resolve()


def binary_exec(spec: dict) -> Path | None:
    """Resolved executable for a `launch: "binary"` backend (e.g. LichtFeld Studio),
    or None if missing / not executable. Such backends invoke a compiled binary
    directly instead of a conda-env python (so env_python / torch checks don't apply)."""
    exe = Path(spec.get("exec", "")).expanduser()
    if not exe.is_absolute():
        exe = (BASE / exe).resolve()
    return exe if (exe.is_file() and os.access(exe, os.X_OK)) else None


# --------------------------------------------------------------------------- #
# Backend listing (built-ins merged with per-machine backends.json)
# --------------------------------------------------------------------------- #
def load_backends() -> dict[str, dict]:
    out = {k: dict(v) for k, v in BUILTIN_BACKENDS.items()}
    if BACKENDS_FILE.is_file():
        try:
            user = json.loads(BACKENDS_FILE.read_text())
            for name, spec in (user or {}).items():
                if name.startswith("_") or not isinstance(spec, dict):
                    continue    # skip comment / non-backend keys (e.g. "_comment")
                out[name] = {**out.get(name, {}), **spec}     # shallow override
        except Exception:
            pass            # a broken backends.json must not take the panel down
    return out


def get_backend(name: str) -> dict | None:
    return load_backends().get(name)


def available_backends() -> list[dict]:
    """Light summary for the UI form: each backend with a resolved `ready` flag."""
    res = []
    for name, spec in load_backends().items():
        if spec.get("launch") == "binary":            # compiled trainer (LichtFeld)
            ready = bool(binary_exec(spec))
        else:
            py = env_python(spec)
            repo = repo_path(spec)
            ready = bool(py) and (repo / spec.get("train_script", "train.py")).is_file()
        res.append({
            "name": name,
            "label": spec.get("label", name),
            "ready": ready,
            "binary": spec.get("launch") == "binary",  # compiled trainer (LichtFeld): no conda/extra
            "params": spec.get("params", []),         # training-param schema
            # mesh extraction is backend-specific (e.g. GS-2M's render.py); a backend
            # supports it only if it declares mesh_args. gsplat & co. won't.
            "mesh": bool(spec.get("mesh_args")),
            "mesh_params": spec.get("mesh_params", []),
            # marker-board scaling: a backend opts in by declaring a board geometry
            # (marker_defaults). The scripts are panel-owned (tools/) and run in
            # this backend's env. The UI shows a checkbox + the configured spec.
            "marker": bool(spec.get("marker_defaults")),
            "marker_defaults": spec.get("marker_defaults", {}),
        })
    return res


# --------------------------------------------------------------------------- #
# GPU discovery (never hardcode the GPU count — machines differ)
# --------------------------------------------------------------------------- #
def list_gpus() -> list[dict]:
    out = probe(["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "mem_mb": int(parts[2]) if parts[2].isdigit() else 0})
    return gpus


# --------------------------------------------------------------------------- #
# Preflight /doctor — turn "deploy to a new machine" into a checklist
# --------------------------------------------------------------------------- #
def _colmap_check() -> dict:
    """The pre-`system` report shape, kept for /api/doctor consumers. Derived from
    the system check so there's exactly one implementation of the probing."""
    c = colmap_check()
    ok = c["status"] == "ok"
    return {"bin": COLMAP_BIN, "path": c["value"] if ok else None,
            "version": c["detail"] or None, "ok": ok}


def _probe_env(py: Path, imports: list[str]) -> dict:
    """Run a tiny script in the trainer env: torch+CUDA + import the compiled
    submodules. Importing e.g. diff_gaussian_rasterization here catches the most
    common move-to-new-machine failure (extension built for another GPU arch)."""
    code = (  # noqa: UP031  (字串內含 {dict};用 % 格式化才不會撞到 .format 的大括號)
        "import json, sys\n"
        "r = {'python': sys.version.split()[0]}\n"
        "try:\n"
        "    import torch\n"
        "    r['torch'] = torch.__version__\n"
        "    r['cuda'] = bool(torch.cuda.is_available())\n"
        "    r['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
        "except Exception as e:\n"
        "    r['torch_error'] = repr(e)\n"
        "fail = {}\n"
        "for m in %r:\n"
        "    if m == 'torch':\n"
        "        continue\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception as e:\n"
        "        fail[m] = repr(e)\n"
        "r['import_fail'] = fail\n"
        "print(json.dumps(r))\n"
    ) % (list(imports),)
    try:
        out = subprocess.check_output([str(py), "-c", code], text=True,
                                      stderr=subprocess.STDOUT, timeout=180)
        return json.loads(out.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "probe timed out (CUDA init >180s?)"}
    except Exception as e:
        return {"error": repr(e)}


def _backend_checks(spec: dict, py: Path | None, repo: Path,
                    script_ok: bool, probe_result: dict | None) -> list[dict]:
    """A Python backend's readiness as uniform check rows.

    The hints live HERE, in the data, not in each renderer: doctor.html and
    doctor_cli.py previously each decided what `env_python is None` means and only
    the CLI ever grew the "how do I fix it" text, so the web page showed a red light
    with no remedy. Both now just iterate.

    Everything is `warn`, never `err`: backends are per-machine opt-in, so a box
    that only runs GS-2M must not be reported as a failed deployment because
    LichtFeld isn't built. The 訓練 tab greys out whatever isn't ready.
    """
    checks = [
        make_check("env_python", "env python", "ok" if py else "warn",
                   str(py) if py else f"conda env '{spec.get('conda_env')}'",
                   "" if py else "找不到這個 env 的 python",
                   "" if py else '建好該 env,或在 backends.json 設絕對路徑 "python"'),
        make_check("repo", "repo / train script", "ok" if script_ok else "warn", str(repo),
                   "" if script_ok else f"找不到 {spec.get('train_script', 'train.py')}",
                   "" if script_ok else 'clone 到面板的兄弟目錄,或在 backends.json 覆蓋 "repo"'),
    ]
    if not probe_result:                 # deep=False, or no interpreter to probe
        return checks

    if probe_result.get("error"):
        checks.append(make_check("probe", "env 探測", "warn", "", probe_result["error"]))
        return checks

    torch_ver, cuda = probe_result.get("torch"), probe_result.get("cuda")
    if torch_ver:
        device = probe_result.get("device")
        checks.append(make_check(
            "torch", "torch", "ok" if cuda else "warn",
            f"{torch_ver} · CUDA {'可用' if cuda else '不可用'}"
            f"{' · ' + device if device else ''}",
            "" if cuda else "torch 看不到 CUDA,訓練會失敗",
            "" if cuda else "確認驅動版本與這個 env 的 torch cu 版本相符"))
    else:
        checks.append(make_check("torch", "torch", "warn", "",
                                 probe_result.get("torch_error", "無法載入 torch")))
    # Importing the compiled extensions is what catches the most common
    # move-to-a-new-machine failure, so give it its own row either way. A probe
    # that got this far always has import_fail (a dict, empty when all imports ok).
    failed = probe_result.get("import_fail") or {}
    if failed:
        checks.append(make_check(
            "submodules", "編譯子模組", "warn", ", ".join(failed),
            "; ".join(f"{mod}: {err}" for mod, err in failed.items()),
            "通常是 extension 是為別張顯卡的 arch 編的 — 在這台機器重編"))
    else:
        checks.append(make_check("submodules", "編譯子模組", "ok", "全部 import 成功"))
    return checks


def doctor(deep: bool = True) -> dict:
    """Full preflight report. deep=True imports torch in each env (slow, ~seconds
    per backend, triggers CUDA init); deep=False only checks paths exist."""
    report: dict = {
        "colmap": _colmap_check(),
        "conda_envs_dir": str(conda_envs_dir() or ""),
        "gpus": list_gpus(),
        # Host-level checks (ffmpeg + its blurdetect filter, disks, cudnn, gsutil,
        # SuperSplat, local.env sanity) — see pipeline/preflight.py for the shape.
        "system": system_report(deep),
        "backends": {},
    }
    for name, spec in load_backends().items():
        if spec.get("launch") == "binary":            # compiled trainer (LichtFeld)
            exe = binary_exec(spec)
            exec_path = str(Path(spec.get("exec", "")).expanduser())
            item = {
                "label": spec.get("label", name),
                "launch": "binary",
                "exec": exec_path,
                "exec_ok": bool(exe),
                "config": spec.get("config", ""),
                "mesh": bool(spec.get("mesh_args")),
                "ready": bool(exe),
                "checks": [make_check(
                    "exec", "exec", "ok" if exe else "warn", exec_path,
                    "" if exe else "找不到編譯好的執行檔",
                    "" if exe else '在 ../LichtFeld-Studio 建置,或在 backends.json 覆蓋 "exec"')],
            }
            report["backends"][name] = item
            continue
        py = env_python(spec)
        repo = repo_path(spec)
        script_ok = (repo / spec.get("train_script", "train.py")).is_file()
        item = {
            "label": spec.get("label", name),
            "conda_env": spec.get("conda_env"),
            "env_python": str(py) if py else None,
            "repo": str(repo),
            "repo_ok": script_ok,
            "ready": bool(py) and script_ok,
        }
        probe_result = _probe_env(py, spec.get("probe_imports", ["torch"])) if deep and py else None
        if probe_result is not None:
            item["probe"] = probe_result
        item["checks"] = _backend_checks(spec, py, repo, script_ok, probe_result)
        report["backends"][name] = item

    # Depth/normal tool: LichtFeld-Studio's own `preprocess` subcommand (MoGe-2
    # ONNX, self-downloading) — same compiled binary as the lichtfeld-* trainers,
    # no separate conda env.
    lichtfeld_exe = binary_exec({"exec": BUILTIN_BACKENDS["lichtfeld-mrnf"]["exec"]})
    report["depth"] = {
        "exec": str(Path(BUILTIN_BACKENDS["lichtfeld-mrnf"]["exec"]).expanduser()),
        "exec_ok": bool(lichtfeld_exe),
        "ready": bool(lichtfeld_exe),
    }
    return report
