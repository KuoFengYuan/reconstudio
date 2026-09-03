# Recon Studio

把**影片或照片變成 3D 模型**的本機網頁面板,一條龍跑完:

```
影片 ──抽幀──► 清晰照片 ──COLMAP──► 相機位姿+點雲 ──訓練──► 3DGS 模型 ──Mesh──► 三角網格(.ply)
```

打開瀏覽器就能操作:每一步都有表單、即時 log、可取消、跑完有「接著跑下一步」按鈕自動帶路徑。
也內建瀏覽器 3D 檢視器(點雲 / Mesh / 量尺)、去背編輯器、GCS 雲端搬檔。

**讀這份文件的方式**:

| 你是誰 | 看哪裡 |
|--------|--------|
| 第一次裝機 | [一、安裝](#一安裝第一次部署) |
| 每天操作的人 | [二、使用](#二使用) |
| 要調效能 / 接雲端 / 用 GPS 航拍 | [三、進階](#三進階) |
| 要改 code | [四、開發者](#四開發者) |

---

# 一、安裝(第一次部署)

> 面板本身很輕(純 Python、不含 torch);重的是 GPU 訓練環境。任何時候都可以用
> **`./run.sh --doctor`**(終端機)或 **`/doctor`** 頁面逐項檢查,紅燈變綠燈再開始用。

**下面的節次就是執行順序**,由快到慢:

| # | 做什麼 | 大概要多久 |
|---|--------|-----------|
| [0](#0-系統前置) | 系統前置:git / conda / NVIDIA 驅動 | 10 分鐘 |
| [1](#1-外部工具) | 外部工具:colmap + ffmpeg | 5 分鐘(apt)~ 1 小時(自己編) |
| [2](#2-面板本體) | 面板本體:`./setup.sh` | 2~3 分鐘,**全自動** |
| [3](#3-訓練後端需要-gpu每台機器各自編譯) | 訓練後端(每台機器各自編譯) | 20 分鐘 ~ 數小時 ← **最花時間** |
| [4](#4-選用元件) · [5](#5-這台機器的設定) | 選用元件、微調設定 | 看需求 |
| [6](#6-環境檢查) · [7](#7-啟動) | 健檢 + 啟動 | 1 分鐘 |

> 第 2 步的 `setup.sh` 結尾會跑一次健檢。那時第 1、3 步還沒做完,**紅燈是正常的**
> ——它就是在告訴你還缺什麼。

## 0. 系統前置

```bash
sudo apt install -y git build-essential
nvidia-smi                       # 驅動要先裝好,看得到卡才有得訓練
```

還沒有 conda 的話裝一個(`setup.sh` 找不到 conda 會直接停下來):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 1. 外部工具

| 工具 | 用途 | 注意 |
|------|------|------|
| **colmap**(3.x / 4.x) | 重建核心 | 不在 `PATH` 就在 `local.env` 設 `COLMAP_BIN` |
| **ffmpeg** | 抽幀去模糊、縮圖 | **必須含 `blurdetect` filter**;有 NVDEC build 可 GPU 解碼 |
| **exiftool**(選用) | undistort 前消毒 Canon 空字串 EXIF | 沒裝也能跑,只是踩到才知道(見下) |

多數情況 apt 版就夠:

```bash
sudo apt install -y colmap ffmpeg libimage-exiftool-perl
ffmpeg -hide_banner -filters | grep blurdetect   # 必須有這行,否則抽幀會失敗
```

> **exiftool 是做什麼的**:Canon 機身沒設定版權資訊時,仍會把 `Artist`/`Copyright`
> 寫成空字串(而不是完全不寫)。COLMAP 的 `image_undistorter` 重新編碼這種影像時,
> OpenImageIO 2.4.17 的 IPTC 編碼器會 assert 崩潰(`SIGABRT`),整個 undistort 階段
> 直接失敗。裝了 exiftool,pipeline 會在每次 undistort 前自動清掉這兩個空字串欄位
> (只在欄位是空字串時才動手,對其他相機是完全無感的 no-op,而且只改 metadata、
> 像素不變)。沒裝 exiftool 不影響大多數素材,只是 Canon 拍的資料集有機率在
> undistort 卡死,`./run.sh --doctor` 會提醒。

> **什麼時候需要自己編 colmap**:要用 Caspar BA(`MAPPER=global-caspar` 後端)、或
> 要讀航拍大圖的 IPTC/大小寫副檔名時,apt 版不夠 —— 得從源碼編。編完把路徑寫進
> `local.env` 的 `COLMAP_BIN`(或 `sudo cmake --install build` 裝到 `PATH`)。

## 2. 面板本體

```bash
git clone https://github.com/KuoFengYuan/reconstudio.git
cd reconstudio
./setup.sh          # 建 conda env + 裝依賴 + 產生 local.env + 跑環境檢查
```

`setup.sh` 會偵測這台機器的 conda 位置、最大的非 root 磁碟、有沒有 NVDEC 版 ffmpeg,
產生 `local.env`,最後印出環境檢查告訴你還缺什麼。**通常不需要再手改。**
可重複執行:env 已存在就只更新套件,`local.env` 已存在**絕不覆蓋**(偵測結果會另存成
`local.env.detected` 讓你自己比對)。

> 產生的 `local.env` 裡,磁碟/ffmpeg 那幾行是**註解掉的**——這是刻意的。`run.sh` 每次啟動
> 都會重跑同一份偵測,把值寫死反而會**遮蔽偵測**:之後裝了 NVDEC ffmpeg、換了資料磁碟,
> run.sh 都不會發現。要固定某一項就把該行的 `#` 拿掉。(`CONDA_ROOT` 是例外,寫死可以
> 省下每次啟動一次 `conda info --base`,約 0.5 秒。)

<details><summary>不想用腳本 / 想自己一步步來</summary>

```bash
conda create -n rec python=3.10 -y
conda run -n rec pip install -r requirements.txt
cp local.env.example local.env    # 再自己填路徑
./run.sh --doctor                 # 確認缺什麼
```

`--env NAME` 換 env 名、`--skip-env` 只重新產生設定不碰 conda。
</details>

## 3. 訓練後端(需要 GPU,每台機器各自編譯)

後端是「設定」不是「程式」:加一個訓練器 = 改 `backends.json`,不用改 code。內建預設 GS-2M。

**GS-2M(預設,訓練 + mesh 都有)** — 放在面板的兄弟目錄 `../GS-2M`、conda env 名 `gs2m`,
名字路徑都對就零設定:

```bash
cd ..                                    # 面板 repo 的上一層
git clone https://github.com/ndming/GS-2M.git
cd GS-2M
conda env create --file environment.yml  # 建 env "gs2m"
conda activate gs2m
pip install -r requirements.txt          # 編 CUDA submodule(依顯卡,要跑一陣子)
```

> 要量**實際尺寸 (mm)** 再補:`conda run -n gs2m pip install opencv-contrib-python plyfile`

> **新顯卡(Blackwell 等)注意**:GS-2M 的 `environment.yml` 釘的 torch 版本可能沒有
> 你這張卡的 CUDA arch。裝完先確認 `./run.sh --doctor` 的 gs2m 那項 torch/CUDA 是綠的
> ——它會真的在 gs2m env 裡 import 編譯出來的 CUDA submodule,這是「換機器後最常壞掉」
> 的地方(extension 是為別張卡的 arch 編的)。需要時改用對應的 cu12x wheel 重編。

**LichtFeld Studio(選用,只訓練、不 mesh)** — C++/CUDA binary,需 CUDA 12.8+ 與 vcpkg:

```bash
sudo apt install -y libcudnn9-cuda-12   # onnxruntime 的 CUDA provider 需要,缺了會跑到一半才爆
cd .. && git clone https://github.com/MrNeRF/LichtFeld-Studio.git && cd LichtFeld-Studio
cmake -B build && cmake --build build -j"$(nproc)"
```

> `libcudnn9` 同時也是 🌊 深度 工具(共用同一份 binary)的必要條件。裝在非標準位置時
> 用 `local.env` 的 `LD_LIBRARY_PATH` 指過去;`./run.sh --doctor` 有專門一項在檢查它。

LichtFeld 的 backend(MR-NF)已內建在 `pipeline/backends.py`,只要
`LichtFeld-Studio` 跟 `reconstudio` 同一層(就是上面 `cd ..` clone 的位置),完全不用
碰 `backends.json`,程式會自動找到 `build/LichtFeld-Studio`。只有這台機器的 build
放在別的地方(不同硬碟、NFS 路徑…)時,才需要在**這台機器自己的** `backends.json` 補一個
`"exec"` 覆蓋(範本見 `backends.example.json`)——其餘參數仍吃內建預設,以後新增參數只要
`git pull` 就全機器同步,不用手動改 JSON。

## 4. 選用元件

- **☁️ GCS 雲端搬檔**:需 Google Cloud SDK + 登入,見[三、進階 — GCS 設定](#gcs-設定一次性)。
- **🧹 SuperSplat(去背 + 點雲檢視)**:**免裝** — `run.sh` 啟動時自動同步到最新版
  (背景跑、不擋啟動;離線就沿用現有版本)。詳見[三、進階](#supersplat-自動更新)。
- **🌊 深度/法向量(選用)** — 為照片產生深度圖和/或法向量圖,給 LichtFeld 訓練做深度/法向量
  監督。有**兩種引擎**,寫出的 `depth/`、`normals/` 完全同格式,下游訓練不需要知道是哪一個
  產生的,可以混用或彼此覆蓋重跑:

  | 引擎 | 需要裝什麼 | 速度(1600px) | 特性 |
  |---|---|---|---|
  | **LichtFeld preprocess(MoGe-2)**(預設) | 免裝,共用訓練 backend 的 binary | ~40 ms/張 | 內建 MoGe-2 ViT-B,第一次執行自動下載權重,不需 torch |
  | **MoGe-3(PyTorch)** | 需 `moge3` conda env(見下) | ~0.5 s/張 | 細節明顯較佳,法向量差距最明顯 |

  MoGe-2 這條**免額外裝 conda env** — 直接跑 LichtFeld-Studio 自己編譯出來的 `preprocess`
  子指令(跟 lichtfeld-mrnf 訓練 backend 共用同一份 binary)。只要 `../LichtFeld-Studio`
  建置好(見上面「3. 訓練後端」),開 `/doctor` 就會看到「深度/法向量生成」變綠燈。

  **MoGe-3 引擎安裝**(選用;想要更銳利的法向量再裝):

  ```bash
  conda create -y -n moge3 python=3.11
  conda run -n moge3 pip install torch "git+https://github.com/microsoft/MoGe.git"
  # 驗證:應印出 True 和 OK
  conda run -n moge3 python -c \
    "import torch; print(torch.cuda.is_available()); \
     from moge.model.v3 import MoGeModel; print('OK')"
  ```

  幾個要點:

  - **env 名字要正好是 `moge3`** — 面板用 `conda_env: "moge3"` 找它(跟訓練 backend 同一套
    解析邏輯)。裝在別處就在 `backends.json` 覆寫 `python`。
  - **要獨立的 env,不要塞進訓練 env** — MoGe 會固定自己的 torch 版本,而面板本體的 env
    根本沒有 torch。
  - 上面刻意**不指定 torch 的 CUDA 版本**,讓 pip 自己解。若你的網路有 SSL 攔截,
    `download.pytorch.org` 那個 index 可能因為憑證鏈被拒,走預設 PyPI 反而會過。
  - 權重(預設 `Ruicheng/moge-3-vitl`,370M)第一次執行時從 HuggingFace 自動下載。
  - **需要 CUDA**;沒有可用 GPU 時這條引擎直接報錯結束,不會退回 CPU 慢跑。
  - LichtFeld **無法**載入 MoGe-3:它是把 MoGe-2 的架構手刻成 C++/CUDA(沒有 ONNX
    runtime),`--model` 只換權重不換架構,所以 MoGe-3 只能走這條獨立的 PyTorch 路徑。

  裝好後面板「深度/法線生成」的**引擎**選單就會出現 MoGe-3。用法見
  [二、使用 — 🌊 深度](#-深度影像--深度圖法向量圖--訓練深度法向量監督)。

### 從舊機器搬過來可以省的

已經有一台裝好的機器時,這兩樣**直接複製比重建快**:

```bash
# SuperSplat bundle:省掉第一次啟動的背景建置,新機器也就不需要 node/npm
rsync -a 舊機器:~/repo/reconstudio/static/supersplat/ static/supersplat/
```

`local.env` **不要整份複製** —— 路徑、磁碟、port 都是機器專屬的。跑 `./setup.sh` 讓它重新
偵測,只把真正跨機器通用的那幾行搬過去(例如 `CLOUDSDK_CORE_PROJECT`)。

## 5. 這台機器的設定

`setup.sh` 已經產生 `local.env`(路徑 / port / binary 位置),要調再開它——每個選項的
完整說明在 `local.env.example`。

`backends.json` **通常不用建**:內建預設已涵蓋「兄弟目錄 + 標準 env 名」的情況。只有這台
機器的 repo/build 放在別處才需要,而且只寫要覆蓋的那幾個 key(範本見 `backends.example.json`)。

> `local.env` 裡打錯字或留著失效的舊變數**不會報錯**——`Settings` 會靜默忽略它、改用預設值。
> `./run.sh --doctor` 的「local.env 變數」一項專門抓這個。

**`setup.sh` 猜不到、要你自己判斷的一項**:`COLMAP_PANEL_RESIZE_WORKERS`(COLMAP FullHD
縮圖的平行 ffmpeg 數)取決於**來源檔放在哪種磁碟**,不是 CPU 核數。單顆 HDD 上開太多
worker 會讓磁碟一直隨機尋軌、CPU 空等而更慢(大檔如 102MP 航拍 TIFF 的甜蜜點約 4~6);
放 SSD/NVMe 就可以往上加。預設是 CPU 核數(上限 32),對 HDD 來說通常太高。

## 6. 環境檢查

```bash
./run.sh --doctor          # 終端機版:colmap / ffmpeg / exiftool / 磁碟 / cudnn / 後端 / GPU 全部逐項
./run.sh --doctor --fast   # 跳過每個後端的 torch/CUDA 探測(快)
./run.sh --doctor --json   # 給腳本吃的 JSON
```

必要條件全通過就 **exit 0**,可以直接串在部署腳本後面。它檢查的不只是「檔案在不在」,還包括
幾個典型的**靜默失敗**:ffmpeg 有沒有編進 `blurdetect` filter、設定的 hwaccel 這個 build
到底支不支援、資料/暫存磁碟是真的可寫還是唯讀掛載、`libcudnn.so.9` 解析得到嗎
(LichtFeld 的 onnxruntime 在跑到一半才會爆)。

`WARN` 代表「某個**選用**功能不能用」,不算部署失敗;`FAIL` 才是一定會出問題。
同一份報告的網頁版在 `/doctor`。

## 7. 啟動

```bash
./run.sh        # → 開瀏覽器 http://127.0.0.1:8077
```

遠端機器用 `ssh -L 8077:127.0.0.1:8077 user@host`,或 `HOST=0.0.0.0 ./run.sh`。

---

# 二、使用

## 介面總覽

左側分三排:**功能**(重建流程的五站)、**工具**(獨立小工具,跟流程無關)、
**檢視**(看檔案的檢視器)。各分頁互相獨立,**不強制照順序**——但每站跑完都有
「接著跑下一步」按鈕幫你帶好路徑。

| 排 | 分頁 | 做什麼 | 輸入 → 輸出 |
|----|------|--------|------------|
| 功能 | ☁️ 資料 | GCS 雲端 ⇄ 本機搬檔 | gs:// ⇄ 本機資料夾 |
| 功能 | 抽幀 | 影片抽成清晰照片(去模糊) | 影片資料夾 → 每支影片一夾 .jpg |
| 功能 | COLMAP | 照片算相機位姿 + 稀疏點雲 | 照片 → workspace(sparse + 去畸變 dense) |
| 功能 | 訓練 | 3DGS 訓練(可開深度監督) | COLMAP workspace → 3DGS 模型 |
| 功能 | Mesh | 模型抽三角網格,可量實際尺寸 | 3DGS 模型 → `tsdf_post.ply`(+mm 版) |
| 工具 | 🌊 深度 | 為照片產生深度/法向量圖(LichtFeld preprocess) | 影像夾 → `depth/`、`normals/`(同名同尺寸) |
| 工具 | 🧱 分塊 | 大場景切成可獨立訓練的子塊 | COLMAP workspace → 每塊一夾 |
| 檢視 | 👁 Mesh Viewer | 看任意 mesh 檔(不綁 job) | `.ply/.obj/.stl/.glb`,server 或本機檔 |
| 檢視 | ✨ SuperSplat | 看任意點雲 / 3DGS(不綁 job) | `.ply/.splat/.ksplat/.spz/.sog` |

右側是**執行 log 與歷史**:即時 log、階段 stepper、可取消;歷史可篩選 / 搜尋 / 刪除。

## 抽幀(影片 → 清晰照片)

選影片資料夾 → `out_dir` 自動帶出 → 設每秒抽幾張(`fps`)和去模糊強度(留前 `keep%` 最清晰的,
或固定閾值)→ **▶ 抽幀 + 去模糊**。GPU 解碼自動啟用,失敗的影片自動退回 CPU。

```
輸入  <root>/<group>/<video>.MOV          例  FY115/FY115_0518/A/IMG_3600.MOV
輸出  <out>/<group>/frames_<video>/*.jpg  例  FY115/0518_colmap/A/frames_IMG_3600/*.jpg
```

## COLMAP(照片 → 重建)

設 `image_root`(照片)和 `workspace`(輸出)。兩個常用選項:

- **影像解析度**:預設把每張縮成長邊 ≤1920 的實體副本(`workspace/images_1920/`)再跑整條
  COLMAP——4K / 航拍原圖直接跑又慢又容易出問題。要高解析訓練圖可選 2560 / 4096;
  「保持原樣」用原始檔。(Canon 空字串 EXIF 導致 undistort 崩潰的問題,現在不論選哪個
  解析度都會在 undistort 前自動消毒,不用靠縮圖重編碼側面繞過——見[一、安裝 — 外部工具](#1-外部工具)的 exiftool 說明。)
- **版面(layout)**:自動偵測;`single` = 一夾照片一台相機、`multi` = 每個子資料夾一台相機、
  `nested` = 抽幀輸出的兩層結構。

**▶ 啟動 COLMAP** → log 會顯示偵測到的 layout 和各階段進度。跑完按 **🧊 檢視 3D 結果**
檢查品質,或 **🧠 接著訓練**。

> 航拍 / RTK 有 GPS 的資料,展開「GPS 對齊 + 大場景」可以更快更穩,見
> [三、進階 — GPS](#gps--大場景航拍-rtk)。

## 訓練(重建 → 3DGS 模型)

選 backend(環境沒裝好會灰掉,旁邊有 /doctor 連結)、`source`(COLMAP workspace)、
`model_path`(輸出位置)、**GPU 編號**。參數依 backend 動態顯示,每個欄位都有說明。
右側狀態列會顯示 `iter N/total` 和 loss。

> 訓練只吃**去畸變(PINHOLE)**模型——接錯會在開跑前直接報錯,不會白跑。
> 原 COLMAP workspace 完全不動(symlink 進去)。

### 開啟深度/法向量監督(depth / normal loss)

LichtFeld backend(MR-NF)的參數區有深度與法向量兩組選項,先用「🌊 深度」產生對應
的圖後即可開啟:

- **深度損失 (use-depth-loss)** — 勾選才啟用(預設關)。深度圖沒有自動生成,一定要先用
  「🌊 深度」工具產生 `depth/`。
- **深度損失模式** — `ssi`(**自動偵測,預設,建議**)、`ssi-disparity`、`ssi-depth`(強制指定
  先驗是反深度或正深度)。
- **深度損失權重** — 預設 `2.0`。
- **法向量損失 (use-normal-loss)** — 勾選才啟用(預設關)。**這版起法向量圖不是必須預先生成**:
  訓練時若 `normals/` 缺圖或尺寸不符,LichtFeld 會自動用內建 MoGe-2 即時補生成(`--no-normal-auto-generate`
  可關掉這個行為)。「🌊 深度」工具仍然有用——可以用原始解析度先跑一次、重複利用、離線檢查——但不再是
  開啟法向量損失的前提。
- **法向量先驗權重** — 預設 `0.005`。
- **深度-法向量一致性權重** — 預設 `0.001`。
- **扁平化權重** — 預設 `0`(法向量監督期間讓高斯最短軸攤平;預設不啟用)。
- **法向量先驗座標系** — `auto`(**自動偵測,建議**)、`camera-opencv`、`camera-opengl`、`world`。

> 前提:`depth/`、`normals/` 要和**這次訓練實際用的 `images/` 同層、同名同尺寸**。LichtFeld 看到
> `--use-depth-loss` / `--use-normal-loss` 會自動掃 `<資料夾>/depth`、`<資料夾>/normals` 對應
> (不用手動指路徑)。

### 開啟 16-bit 色彩訓練(HDR 素材)

LichtFeld backend(MR-NF)參數區有 **16-bit 色彩訓練**(`--use-16bit`)勾選框,預設關。

- 只在來源影像**本身就是 16-bit**(RAW 轉出的 TIFF/PNG、HDR 合成素材)才有意義——一般手機
  /相機直出的 8-bit JPEG/PNG 開這個沒效果,因為動態範圍在源頭就已經被裁掉了。
- 開啟後 LichtFeld 會自動用**無損 JPEG2000** 做磁碟快取(不用另外設定),換取更完整的亮部/
  暗部細節,代價是快取檔案較大、稍慢。

## 🌊 深度(影像 → 深度圖/法向量圖 → 訓練深度/法向量監督)

為每張照片產生深度圖和/或法向量圖,輸出成 **LichtFeld 格式的 `depth/`、`normals/` 資料夾**
(與來源**同名同尺寸**的 PNG),給訓練的[深度/法向量監督](#開啟深度法向量監督depth--normal-loss)
讀取。兩種引擎可選,安裝需求見[一、安裝 — 選用元件](#4-選用元件)。

1. **`images`** — 選照片資料夾(或含 `images/` 的 COLMAP workspace)。
2. **引擎** — `LichtFeld preprocess(MoGe-2)`(**預設**,免裝、~40 ms/張)或
   `MoGe-3(PyTorch)`(細節較佳、~0.5 s/張,需 `moge3` conda env)。
3. **產生內容** — `both`(深度+法向量,**預設**)、`depth`(只有深度)、`normal`(只有法向量)。
4. 輸出固定在資料集根目錄下的 `depth/`、`normals/`(不可自訂位置),跟 LichtFeld 自動掃描
   的路徑一致。
5. 進階(依所選引擎顯示不同欄位):
   - 兩者共用:**bit-depth**(輸出 PNG 位元深度,留空用內建預設 16——8-bit 深度先驗量化
     較明顯)、**覆蓋已存在**(預設略過、可續跑)。
   - 只有 MoGe-2:**model**(自訂 ONNX 模型路徑,留空 = 自動下載官方 MoGe-2 ViT-B)、
     **max-side**(推論最長邊,留空用內建預設 518)。
   - 只有 MoGe-3:**MoGe-3 模型**(HuggingFace id 或本機 `.pt`,留空 = `Ruicheng/moge-3-vitl`;
     `moge-3-vitg` 1.25B 更好但慢很多、吃更多顯存)。MoGe-3 自己會把輸出上採樣回原圖尺寸,
     所以沒有 max-side。
6. **▶ 產生深度/法向量圖** → 右側顯示進度(兩種引擎共用同一組進度解析)。

**兩種引擎要選哪個**:兩者的深度整體排序幾乎一致(實測同一批空拍圖 Spearman +0.995),
差別在細節 —— MoGe-3 能解出梯田邊界、屋頂、電塔等結構,MoGe-2 在這些地方偏糊,法向量圖
的差距比深度圖明顯得多。整批照片量大又只需要深度時 MoGe-2 快得多;要吃法向量監督、或場景
本身結構細碎時再換 MoGe-3。兩者輸出可互相覆蓋,所以可以先用 MoGe-2 全跑一遍,之後只對
關鍵資料集用 MoGe-3 重跑(記得勾**覆蓋已存在**)。

> ⚠️ **像素要對齊**:深度/法向量圖是逐像素對應影像的,所以 `images` 要指到**和訓練 source
> 相同的影像**。若用 COLMAP 去畸變後的 workspace 訓練,就對那個去畸變的 `images/` 產生;
> 若直接用原圖訓練,就對原圖產生。產生後同層會多 `depth/`、`normals/`,**原圖完全不動**。

接著到[訓練](#訓練重建--3dgs-模型)勾「深度損失」/「法向量損失」即可。

## 🧹 去背(選用)

訓練完按 **🧹 在 SuperSplat 去背景** → 內嵌編輯器載入訓練好的點雲:

1. 框選物件 → **Ctrl+I** 反選 → **Delete** 刪背景(**Ctrl+Z** 還原)
2. 按 **✅ 送回去背點雲**

原模型不動。GS-2M 會衍生 `_edited_<時間>` 目錄並自動帶入 Mesh 表單(用乾淨點雲重抽 mesh);
LichtFeld 則直接下載乾淨點雲。

## Mesh(模型 → 三角網格)

選 `model_path`、GPU、TSDF 參數(預設值通常夠用)。要**實際尺寸 (mm)** 就勾「提供 marker」
(拍攝時放 ChArUco 板,板規格已寫在 `backends.json`,免手填)。完成後可下載原始版和 mm 版,
或 **🧊 檢視 Mesh**。

## 檢視器

- **🧊 3D 結果**(COLMAP 後):拖曳旋轉 · 滾輪縮放 · 右鍵平移 · WASD 飛行;雙擊相機看該張
  影像與品質分數。可**框選壞相機移除**、**框選 / 筆刷刪雜點**(先標紅預覽再確認)——都是
  非破壞性,寫到 `cleaned/<時間>/` 新資料夾,可直接拿去重新訓練。
- **🧊 Mesh 檢視**(Mesh job 後):實體打光、mm / recon 切換、**📏 量尺**點兩點量距離、
  線框 / 頂點色 / 白底。
- **👁 Mesh Viewer(工具)**:不綁 job,看任何 mesh 檔。填 server 路徑(「瀏覽」逐層挑檔),
  或**留空直接開**,進去選這台電腦的檔案 / 直接拖放(瀏覽器內解析,不上傳)。
  格式不符或檔案壞掉會直接顯示原因。
- **✨ SuperSplat(工具)**:同樣兩種開法,專看點雲 / 3DGS(訓練輸出的 `point_cloud.ply`、
  `.splat` 等)。看 mesh 用 Mesh Viewer,看點雲用這個。

---

# 三、進階

## GPS / 大場景(航拍 RTK)

輸入若**每張**都有 EXIF GPS(JPEG、TIFF 都支援;影片幀沒有 GPS),可解鎖:

| 選項 | 作用 | 主要參數 |
|------|------|---------|
| `MATCHER=spatial` | 只比對 GPS 鄰近的影像,大場景快且穩 | `SPATIAL_MAX_NEIGHBORS` · `SPATIAL_MAX_DISTANCE`(m) |
| `MAPPER=pose_prior` | GPS 先驗進 BA,抗漂移、輸出直接公制 | `PRIOR_STD_X/Y/Z`(GPS 精度 m;消費級 3~5、RTK ~0.02) |
| `GPS_ALIGN` | 事後把模型對齊到 ENU 公尺座標 | `GPS_ALIGN_MAX_ERROR`(m) |

- 勾任一 GPS 選項時,開跑前會檢查 **100% GPS 覆蓋**,不足直接中斷(缺 GPS 的那張無法定位)。
- 縮圖不會弄丟 GPS:JPEG 會把 EXIF 接回縮圖;TIFF 的 GPS 由面板直接寫進 COLMAP 資料庫
  (COLMAP 自己讀不到 TIFF GPS)。
- **`REORIENT` × GPS**:沒開 GPS 時用 PCA 猜重力轉正 + 縮放;開了 GPS 則只做 Z-up→Y-up
  軸轉、**保留公尺尺度**。要「真實公尺 + 檢視器裡正立」就 `GPS_ALIGN` 和 `REORIENT` 都勾。

## 環境變數

> **幾乎都有預設,留空即可。** 唯一必填的是用 GCS 時的 `CLOUDSDK_CORE_PROJECT`。
> 全部寫在 `local.env`(`run.sh` 會載入並傳給 ffmpeg / colmap / gsutil)。

| env | default | 什麼時候設 |
|-----|---------|-----------|
| `CLOUDSDK_CORE_PROJECT` | — | **用 GCS 必填**(GCP project id) |
| `COLMAP_BIN` / `FFMPEG_BIN` / `GSUTIL_BIN` | PATH | binary 不在 PATH 時 |
| `RECON_STUDIO_DATA` | `/mnt/ssd1/recon_studio/data` 或 `~/.recon_studio` | job 狀態 + log 的落地磁碟(建議放大碟) |
| `RECON_STUDIO_BROWSE_ROOT` | `/mnt/ssd1` 或 `/` | 「瀏覽」按鈕的根目錄 |
| `RECON_STUDIO_DEST_ROOT` | `/` | GCS 下載落地根目錄 |
| `RECON_STUDIO_GCS_ROOT` | 空(列全部 bucket) | GCS 瀏覽器起始 `gs://` 前綴 |
| `CONDA_ROOT` / `CONDA_ENV` | 自動偵測 / `rec` | conda 不在標準位置時 |
| `FFMPEG_HWACCEL` | `cuda` | 設 `none` 強制 CPU 解碼 |
| `COLMAP_PANEL_MAX_JOBS` | `4` | 同時跑幾個 job |
| `COLMAP_PANEL_RESIZE_WORKERS` | CPU 數(≤32) | 縮圖並行 ffmpeg 數 |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | 綁定位址 |
| `SUPERSPLAT_AUTOUPDATE` | `1` | 設 `0` 關掉 SuperSplat 啟動自動更新 |
| `SUPERSPLAT_VER` | latest | 釘住 SuperSplat 版本(`vX.Y.Z`) |

## GCS 設定(一次性)

```bash
# 1) 裝 Google Cloud SDK:https://cloud.google.com/sdk/docs/install
# 2) 登入(遠端無頭機加 --no-launch-browser;無人值守可用 service account)
gcloud auth login
# 3) 設定預設 project(不設的話列 bucket 會報錯)——二選一:
gcloud config set project <YOUR_PROJECT_ID>
#    或寫進 local.env:CLOUDSDK_CORE_PROJECT=<YOUR_PROJECT_ID>
# 4) 驗證(應列出 buckets)
gsutil ls
```

「☁️ 資料」分頁與重建流程**解耦**:下載用 `gsutil -m rsync`(可續傳、只補差異),
上傳支援多選檔案 / 資料夾。搬完到其他分頁用「瀏覽」選那個資料夾即可。

## SuperSplat 自動更新

`run.sh` 每次啟動會在**背景**檢查 SuperSplat 上游最新版(一次 `git ls-remote`,有新版才重建):

- 換版是**原子的**——建好才瞬間切換,期間舊版照常服務
- **失敗無害**——離線、缺 node、或新版與面板 patch 衝突時,沿用現有版本,
  細節在 `$RECON_STUDIO_DATA/supersplat_build.log`
- 手動重建:`FORCE=1 SUPERSPLAT_VER=latest ./tools/build_supersplat.sh`(需 node ≥18 + npm + git)

---

# 四、開發者

三層架構,依賴單向(`app → web/ → pipeline/`,無循環):

```
app.py        app factory:建 FastAPI、mount static、include routers(~30 行)
└─ web/       HTTP 層
   ├─ routers/  pages · browse · create · jobs · viz · viewer · doctor(APIRouter)
   ├─ services/ models(job→路徑解析)· forms(表單→參數驗證)
   └─ shared.py templates / _page / UI 常數
jobs.py       JobManager:asyncio 佇列、N=MAX_JOBS workers、狀態存檔 + log 解析
pipeline/     領域層(torch-free,shell out 到外部工具)
   config.py    Settings:所有設定的單一來源(pydantic-settings)
   runner.py    子行程執行 + 取消        backends.py  後端登錄 + 後端 preflight
   preflight.py 系統層檢查(ffmpeg/磁碟/cudnn/…)  doctor_cli.py  終端機版報告
   frames / colmap/ / train / gcs       model.py     解析 COLMAP 稀疏模型(讀取快取)
```

請求流程:`form ─POST /ui/*─► JobManager(asyncio 佇列)─► run_* 在 thread 內 shell out ─► log ─► 瀏覽器`。
即時更新走每分頁一條 WebSocket(joblist 刷新 + log tail 多工)。job 狀態存
`RECON_STUDIO_DATA/jobs/<id>/`;取消 = `SIGTERM` 子行程群組;COLMAP / 縮圖用 sentinel
做 idempotent(勾 `FORCE` 重跑)。

## 主要檔案

| path | 角色 |
|------|------|
| `web/routers/` | `pages` · `browse`(資料夾/檔案 picker)· `create`(建 job)· `jobs`(查詢/取消/log)· `viz`(3D/mesh 檢視+下載+cull+去背)· `viewer`(獨立 mesh viewer)· `doctor` |
| `web/services/` | `models.py`(job→路徑解析)· `forms.py`(表單→參數驗證) |
| `jobs.py` | `JobManager`(佇列、workers、cancel/delete)+ log 解析 |
| `pipeline/colmap/` | `_run`(orchestrator:stages、sentinels)· `_layout`(版面偵測)· `_resize`(並行縮圖)· `_gps`(EXIF GPS 讀取 JPEG+TIFF、pose prior 注入) |
| `pipeline/frames.py` / `train.py` | 抽幀去模糊 / 訓練 + mesh(+ChArUco mm 縮放) |
| `pipeline/depth.py` / `moge3.py` | 深度/法向量的兩種引擎:LichtFeld `preprocess`(MoGe-2)/ PyTorch MoGe-3;`jobs._run_depth_engine` 依 `engine` 參數分派 |
| `pipeline/moge3_encode.py` | 兩引擎共用的 PNG 編碼(LichtFeld `build_depth_png`/`build_normals_png` 的移植);刻意不含 torch/cv2,好讓 `tools/moge3_preprocess.py` 在 `moge3` env 裡與測試在 CI 裡都能載入 |
| `pipeline/backends.py` | 後端登錄、env/GPU 解析、CLI builder、`doctor()` 總報告 |
| `pipeline/preflight.py` | 系統層檢查(ffmpeg + blurdetect / 磁碟可寫 / cudnn / gsutil / local.env 失效變數);每項回同一個 `{status, label, value, detail, hint}` 形狀,兩個 renderer 共用 |
| `pipeline/doctor_cli.py` | `./run.sh --doctor`:同一份報告輸出到終端機,exit code 反映有無 FAIL |
| `pipeline/model.py` | COLMAP 稀疏模型解析 + PLY 匯出(LRU 快取) |
| `templates/` | `index.html`(表單)+ htmx partials + three.js 檢視器(`viz` / `mesh_viz` / `mesh_view`) |
| `tools/` | `detect.sh`(conda/磁碟/ffmpeg 偵測,`run.sh` 與 `setup.sh` 共用)· `build_supersplat.sh`(自動更新也走這)· `moge3_preprocess.py`(在 `moge3` env 裡跑的 MoGe-3 產生器)· marker 量尺 / 縮放腳本 |
| `tests/` | 離線單元測試(無需 colmap/ffmpeg/GPU/網路) |

## 開發

```bash
pip install -e ".[dev]"   # 面板 + ruff / mypy / pytest
pytest                    # 離線單元測試
ruff check .              # lint
mypy pipeline/config.py   # 型別檢查(目前嚴格守住 config.py)
```

CI 在每次 push / PR 跑 ruff + mypy + pytest。**擴充慣例**:加 endpoint → `web/routers/` 加
handler;加訓練後端 → 改 `backends.json`(不動 code);加設定 → `pipeline/config.py` 加欄位。

> 改 `.py` 需重啟;改 `templates/` 重整頁面即可。

---

# License

Recon Studio 自身的 code 以**非商業、研究與評估用途**釋出([`LICENSE`](LICENSE)),與其核心訓練器
依賴一致。整合的第三方工具各依其授權([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md))——
特別是 **GS-2M / 3D Gaussian Splatting(Inria,非商業)**、COLMAP(BSD)、FFmpeg(LGPL/GPL)。
因此訓練與 mesh 階段為非商業用途;Gaussian-Splatting 技術的商業使用請洽 Inria。
