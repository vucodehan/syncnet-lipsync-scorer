r"""SyncNet lip-sync score for audio-driven takes: LSE-C, LSE-D and the AV offset, as lip-sync papers report them, on the CPU.

    python syncnet_score.py TAKE.mp4 [TAKE2.mp4 ...] [--audio VOCAL.wav] [--offset 0.0] [--threads 8] [-v]
    (CPU torch, python_speech_features, MediaPipe, OpenCV: requirements.txt; ffmpeg on PATH)

Use it to rank takes of the same song section and to pre-screen seeds before a person looks: higher LSE-C (sync
confidence) and lower LSE-D (distance) are better, and offset_frames should match the other takes of the same engine.
Compare takes of the same audio; absolute values move with the song, face size and style. LSE-C gaps under about 0.15
between takes are within the method's noise. Speed (16.7 s take, 8 threads, WanGP rendering alongside): about 20 s
alone, 12 s each in a batch (the next take is tracked while SyncNet runs), peak RAM 0.75 GB.

Validated on 2 Nov 2026 against the unmodified syncnet_python pipeline (S3FD) on 11 sung takes: same ranking (Spearman
0.97), LSE-C within 0.13, offsets one frame earlier (see the AVI note below). Vocal delayed 0.5 s: offset -3 -> -15;
unrelated vocal: LSE-C 4.67 -> 0.59.

Method (syncnet_python, the reference behind Wav2Lip's LSE-C / LSE-D): the video goes to 25 fps (ffmpeg fps filter),
the face is cropped as run_pipeline.py crops it (box half-size s and centre x, y smoothed by a 13-frame median filter;
crop from s above to 1.8 s below the centre and 1.4 s to each side, grey 110 outside the frame, cv2.resize to 224x224,
BGR 0-255); 5-frame windows go through the lip stream and the matching 0.2 s of MFCC (python_speech_features defaults:
13 coefficients at 100 Hz from 16 kHz mono int16) through the audio stream. For each offset from -15 to +15 frames the
distance is averaged over the clip: LSE-D is the minimum, LSE-C the median minus the minimum, offset_frames the offset
at the minimum (reference sign: negative = the mouth moves before the sound). A constant delay within +-15 frames moves
the offset, not LSE-C: the search finds the shifted alignment, in the reference too.

Face: MediaPipe Face Landmarker (face_landmarker.task, downloaded on first run) replaces S3FD and scene detection. The S3FD
box is predicted from the face-mesh box (half-size 1.095 x the mesh half-size, centre 0.022 mesh half-sizes right and
0.119 up), fitted against S3FD on 4608 frames of 11 sung takes (per-take error within 3 % of the box). Frames without
a face take a box interpolated from their neighbours; face_rate is the share of frames with a face.

Frames come straight from the source; the reference also round-trips through AVI, XVID and JPEG. Its AVI step delays
the picture one frame when an mp4 has AAC audio (ffmpeg 9), so the reference run_pipeline reads those takes one frame
later (offset +1) with the same LSE-C and LSE-D.

Output: <video stem>.syncnet.json next to each video ({"lse_c", "lse_d", "offset_frames", "frames", "face_rate",
"audio"}), one line per video and, for several videos, a table sorted by LSE-C. --audio is used for every video (else
each video's own track); --offset is where the videos start inside it (seconds). No temp files: ffmpeg pipes, so
parallel runs never collide. Exit 1 with a one-line reason on failure. CPU only; never touches the GPU.

Weights: models\syncnet\syncnet_v2.model, 54,573,114 bytes,
SHA256 961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442, from
http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model (syncnet_python's download_model.sh). The VGG
server was down on 2 Nov 2026; the file came from the Wayback Machine copy of that URL (2025-07-04, SHA-1 matches the
archive record) and matches ByteDance/LatentSync's auxiliary/syncnet_v2.model. Unmodified syncnet_python with it gives
offset 3, min dist 5.358, confidence 10.081 on the reference example.avi (README: 3, 5.353, 10.021).

Attribution: the SyncNet model definition and scoring follow syncnet_python by Joon Son Chung
(https://github.com/joonson/syncnet_python, MIT License, Copyright (c) 2016-present Joon Son Chung; notice below).
Paper: J. S. Chung and A. Zisserman, "Out of time: automated lip sync in the wild", ACCV 2016 workshops.
"""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "models" / "syncnet" / "syncnet_v2.model"
WEIGHTS_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
WEIGHTS_URLS = ("http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
                "https://web.archive.org/web/20250704214616id_/"
                "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model")
FACE_MODEL = HERE / "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                  "face_landmarker.task")
FPS, SR, VSHIFT, BATCH = 25, 16000, 15, 20
CROP_SCALE, MEDIAN = 0.40, 13
BOX_S, BOX_X, BOX_Y = 1.095, 0.022, -0.119


class Fail(Exception):
    pass


def ffmpeg(args, what):
    with tempfile.TemporaryFile() as err:
        p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], stdout=subprocess.PIPE, stderr=err)
        if p.returncode:
            err.seek(0)
            msg = err.read().decode(errors="replace").strip().splitlines()
            raise Fail("%s: %s" % (what, msg[-1] if msg else "ffmpeg exit %d" % p.returncode))
    return p.stdout


def read_audio(src, offset):
    """16 kHz mono int16 samples, as the reference extracts them (ffmpeg -ac 1 -acodec pcm_s16le -ar 16000)."""
    pre = ["-ss", "%.3f" % offset] if offset else []
    raw = ffmpeg([*pre, "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
                 "no audio in %s" % Path(src).name)
    return np.frombuffer(raw, np.int16)


def video_frames(video, w, h, pix):
    """Frames resampled to 25 fps by the ffmpeg fps filter, one (h, w, 3) uint8 array at a time."""
    vf = "fps=%d" % FPS + ("" if (w, h) == probe(video) else ",scale=%d:%d" % (w, h))
    n = w * h * 3
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0", "-vf", vf,
                              "-f", "rawvideo", "-pix_fmt", pix, "-"], stdout=subprocess.PIPE, stderr=err, bufsize=n)
        try:
            while len(buf := p.stdout.read(n)) == n:
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        except GeneratorExit:
            p.kill()
            raise
        finally:
            p.stdout.close()
            rc = p.wait()
        if rc:
            err.seek(0)
            msg = err.read().decode(errors="replace").strip().splitlines()
            raise Fail("cannot decode %s: %s" % (Path(video).name, msg[-1] if msg else "ffmpeg exit %d" % rc))


_probe = {}


def probe(video):
    if video not in _probe:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0:s=x", str(video)], capture_output=True, text=True)
        try:
            _probe[video] = tuple(int(v) for v in p.stdout.strip().splitlines()[0].split("x")[:2])
        except (IndexError, ValueError):
            raise Fail("no video stream in %s" % Path(video).name) from None
    return _probe[video]


@contextlib.contextmanager
def quiet_stderr():
    """MediaPipe and TFLite log to fd 2 from C++; keep the default output to the result lines."""
    sys.stderr.flush()
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    os.close(null)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def face_track(video):
    """S3FD-style box (x, y, s) per 25 fps frame from MediaPipe face-mesh landmarks; NaN where no face."""
    W, H = probe(video)
    w, h = (W // 2 // 2 * 2, H // 2 // 2 * 2) if max(W, H) > 1280 else (W, H)
    with quiet_stderr():
        return _face_track(video, W, H, w, h)


def _face_track(video, W, H, w, h):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    boxes = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, rgb in enumerate(video_frames(video, w, h, "rgb24")):
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                      i * 1000 // FPS)
            if not res.face_landmarks:
                boxes.append((np.nan, np.nan, np.nan))
                continue
            pts = np.array([(p.x * W, p.y * H) for p in res.face_landmarks[0][:468]])
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            ms = max(x1 - x0, y1 - y0) / 2
            boxes.append(((x0 + x1) / 2 + BOX_X * ms, (y0 + y1) / 2 + BOX_Y * ms, BOX_S * ms))
    return np.array(boxes, float).reshape(-1, 3)


def smooth(boxes):
    """Fill gaps by linear interpolation (the reference interpolates its track), then the reference's medfilt(13)."""
    from scipy.signal import medfilt

    ok = np.isfinite(boxes[:, 0])
    idx = np.arange(len(boxes))
    filled = np.stack([np.interp(idx, idx[ok], boxes[ok, k]) for k in range(3)], 1)
    return np.stack([medfilt(filled[:, k], kernel_size=MEDIAN) for k in range(3)], 1)


def crop(frame, x, y, bs):
    """run_pipeline.py crop_video: pad by bsi with 110, cut [my-bs, my+bs(1+2cs)) x [mx-bs(1+cs), mx+bs(1+cs)), 224x224."""
    cs = CROP_SCALE
    bsi = int(bs * (1 + 2 * cs))
    my, mx = y + bsi, x + bsi
    r0, r1 = int(my - bs), int(my + bs * (1 + 2 * cs))
    c0, c1 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
    H, W = frame.shape[:2]
    r0, r1, c0, c1 = max(r0, 0), min(r1, H + 2 * bsi), max(c0, 0), min(c1, W + 2 * bsi)
    out = np.full((r1 - r0, c1 - c0, 3), 110, np.uint8)
    rr0, rr1, cc0, cc1 = max(r0 - bsi, 0), min(r1 - bsi, H), max(c0 - bsi, 0), min(c1 - bsi, W)
    if rr1 > rr0 and cc1 > cc0:
        out[rr0 + bsi - r0:rr1 + bsi - r0, cc0 + bsi - c0:cc1 + bsi - c0] = frame[rr0:rr1, cc0:cc1]
    return cv2.resize(out, (224, 224))


# syncnet_python (SyncNetModel.py, SyncNetInstance.py) is used under this notice:
#
# Copyright (c) 2016-present Joon Son Chung.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def syncnet():
    import torch
    import torch.nn as nn

    class S(nn.Module):
        def __init__(self, num_layers_in_fc_layers=1024):
            super().__init__()
            self.netcnnaud = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
                nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)), nn.BatchNorm2d(512), nn.ReLU())
            self.netfcaud = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                          nn.Linear(512, num_layers_in_fc_layers))
            self.netfclip = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                          nn.Linear(512, num_layers_in_fc_layers))
            self.netcnnlip = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
                nn.BatchNorm3d(96), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
                nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

        def forward_aud(self, x):
            mid = self.netcnnaud(x)
            return self.netfcaud(mid.view((mid.size(0), -1)))

        def forward_lip(self, x):
            mid = self.netcnnlip(x)
            return self.netfclip(mid.view((mid.size(0), -1)))

    if not WEIGHTS.is_file():
        fetch_weights()
    net = S()
    state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not k.endswith("num_batches_tracked") for k in missing):
        raise Fail("weights don't fit the SyncNet model (%s)" % (unexpected or missing)[0])
    return net.eval()


def fetch_weights():
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    part = WEIGHTS.with_suffix(".part%d" % os.getpid())
    for url in WEIGHTS_URLS:
        try:
            urllib.request.urlretrieve(url, part)
        except OSError:
            continue
        if hashlib.sha256(part.read_bytes()).hexdigest() == WEIGHTS_SHA256:
            part.replace(WEIGHTS)
            return
    part.unlink(missing_ok=True)
    raise Fail("syncnet_v2.model missing and not downloadable (see the docstring for the source)")


def sync_scores(net, crops, pcm):
    """SyncNetInstance.evaluate: offset, LSE-C (conf) and LSE-D (min dist) over +-VSHIFT frames."""
    import python_speech_features
    import torch
    import torch.nn.functional as F

    min_length = min(len(crops), len(pcm) // 640)
    lastframe = min_length - 5
    if lastframe < 2 * VSHIFT:
        raise Fail("too short: %d frames of video with audio" % min_length)
    mfcc = python_speech_features.mfcc(pcm, SR).T.astype(np.float32)
    im_feat, cc_feat = [], []
    with torch.inference_mode():
        for i in range(0, lastframe, BATCH):
            n = min(lastframe, i + BATCH) - i
            clip = torch.from_numpy(np.ascontiguousarray(crops[i:i + n + 4].transpose(3, 0, 1, 2))).float()[None]
            mid = net.netcnnlip(clip)
            im_feat.append(net.netfclip(mid[0, :, :, 0, 0].T))
            cc = np.stack([mfcc[:, 4 * v:4 * v + 20] for v in range(i, i + n)])[:, None]
            cc_feat.append(net.forward_aud(torch.from_numpy(cc)))
        im_feat, cc_feat = torch.cat(im_feat), torch.cat(cc_feat)
        win = 2 * VSHIFT + 1
        padded = F.pad(cc_feat, (0, 0, VSHIFT, VSHIFT))
        idx = torch.arange(len(im_feat))[:, None] + torch.arange(win)[None]
        dists = F.pairwise_distance(im_feat[:, None, :].expand(-1, win, -1).reshape(-1, im_feat.shape[1]),
                                    padded[idx].reshape(-1, im_feat.shape[1])).reshape(-1, win)
        mdist = dists.mean(0)
        minval, minidx = torch.min(mdist, 0)
        conf = torch.median(mdist) - minval
    return VSHIFT - int(minidx), float(conf), float(minval), min_length


def peak_ram_mb():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("rest", ctypes.c_size_t * 7)]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    pmc = PMC(cb=ctypes.sizeof(PMC))
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2 ** 20


def prepare(video, audio, offset, min_face_rate):
    """Audio samples, face track and 224x224 crops of one video (runs ahead of SyncNet on the next video)."""
    t0 = time.perf_counter()
    pcm = read_audio(audio or video, offset if audio else 0.0)
    boxes = face_track(video)
    if not len(boxes):
        raise Fail("no frames decoded from %s" % video.name)
    face_rate = float(np.isfinite(boxes[:, 0]).mean())
    if not face_rate or face_rate < min_face_rate:
        raise Fail("face found in only %.0f %% of frames" % (100 * face_rate))
    t1 = time.perf_counter()
    track = smooth(boxes)
    W, H = probe(video)
    crops = np.empty((len(track), 224, 224, 3), np.uint8)
    n = 0
    for i, frame in enumerate(video_frames(video, W, H, "bgr24")):
        if i < len(track):
            crops[i] = crop(frame, *track[i])
            n += 1
    return pcm, face_rate, track, crops[:n], (t1 - t0, time.perf_counter() - t1)


def score_video(net, video, prepared, audio, offset, write_crops):
    pcm, face_rate, track, crops, (t_track, t_crop) = prepared
    t2 = time.perf_counter()
    off, conf, dist, frames = sync_scores(net, crops, pcm)
    t3 = time.perf_counter()
    if write_crops:
        out = video.with_name(video.stem + ".syncnet-crops.mp4")
        src = ["-ss", "%.3f" % offset, "-i", str(audio)] if audio else ["-i", str(video)]
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "224x224",
                              "-r", str(FPS), "-i", "-", *src, "-map", "0:v", "-map", "1:a:0", "-c:v", "libx264",
                              "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
                             stdin=subprocess.PIPE)
        p.communicate(crops.tobytes())
    r = {"lse_c": round(conf, 3), "lse_d": round(dist, 3), "offset_frames": off, "frames": frames,
         "face_rate": round(face_rate, 3), "audio": str(audio or video)}
    detail = "  track %.1f s, crop %.1f s, syncnet %.1f s; box x %.0f y %.0f s %.0f px" % (
        t_track, t_crop, t3 - t2, *track.mean(0))
    return r, t_track + t_crop + t3 - t2, detail


def main():
    ap = argparse.ArgumentParser(description="SyncNet LSE-C / LSE-D / AV offset of lip-synced takes, on the CPU.")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--audio", type=Path, help="vocal used for every video (default: each video's own audio)")
    ap.add_argument("--offset", type=float, default=0.0, help="where the videos start inside --audio, seconds")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8), help="CPU threads for SyncNet")
    ap.add_argument("--min-face-rate", type=float, default=0.5, help="fail a video with a face in fewer frames")
    ap.add_argument("--crops", action="store_true", help="also write <stem>.syncnet-crops.mp4 (the 224x224 crops)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print phase timings, the mean box and peak RAM")
    a = ap.parse_args()
    for p in [*a.videos, *([a.audio] if a.audio else [])]:
        if not p.is_file():
            sys.exit("syncnet_score: not found: %s" % p)
    if a.offset < 0:
        sys.exit("syncnet_score: --offset must be 0 or more")
    if not FACE_MODEL.is_file():
        part = FACE_MODEL.with_suffix(".part%d" % os.getpid())
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, part)
            part.replace(FACE_MODEL)
        except OSError:
            part.unlink(missing_ok=True)
            sys.exit("syncnet_score: face model missing and not downloadable: %s" % FACE_MODEL_URL)

    import torch
    torch.set_num_threads(max(1, a.threads))
    try:
        net = syncnet()
    except Fail as e:
        sys.exit("syncnet_score: %s" % e)
    results, failed, t_all = [], [], time.perf_counter()
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(prepare, a.videos[0], a.audio, a.offset, a.min_face_rate)
        for k, v in enumerate(a.videos):
            cur = nxt
            if k + 1 < len(a.videos):
                nxt = pool.submit(prepare, a.videos[k + 1], a.audio, a.offset, a.min_face_rate)
            try:
                r, secs, detail = score_video(net, v, cur.result(), a.audio, a.offset, a.crops)
            except Exception as e:  # one bad file must not stop a batch
                failed.append(v.name)
                msg = str(e) if isinstance(e, Fail) else "%s: %s" % (type(e).__name__, e)
                r = {"error": msg, "lse_c": None, "lse_d": None, "offset_frames": None, "frames": 0,
                     "face_rate": None, "audio": str(a.audio or v)}
                print("FAIL %s: %s" % (v.name, msg))
            else:
                results.append((v, r))
                print("%-40s LSE-C %6.3f  LSE-D %6.3f  offset %+3d  frames %d  faces %.2f  %.1f s" % (
                    v.name[:40], r["lse_c"], r["lse_d"], r["offset_frames"], r["frames"], r["face_rate"], secs))
                if a.verbose:
                    print(detail)
            v.with_name(v.stem + ".syncnet.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    if len(results) > 1:
        print("\nrank  LSE-C   LSE-D  offset  faces  video")
        for k, (v, r) in enumerate(sorted(results, key=lambda x: -x[1]["lse_c"]), 1):
            print("%4d  %5.3f  %6.3f  %+6d  %5.2f  %s" % (k, r["lse_c"], r["lse_d"], r["offset_frames"], r["face_rate"],
                                                      v.name))
    if len(a.videos) > 1:
        print("%d videos in %.0f s (%.1f s each)" % (len(a.videos), time.perf_counter() - t_all,
                                                    (time.perf_counter() - t_all) / len(a.videos)))
    if a.verbose:
        print("peak RAM %.0f MB, %d threads, CPU %.0f s" % (peak_ram_mb(), a.threads, time.process_time()))
    if failed:
        sys.exit("syncnet_score: %d of %d videos failed (%s)" % (len(failed), len(a.videos), ", ".join(failed)))


if __name__ == "__main__":
    main()
