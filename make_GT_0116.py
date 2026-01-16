import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import pandas as pd
import json
import os

# ==========================================
# 설정 (본인 경로에 맞게 수정 필수)
# ==========================================
DATA_PATH = r"C:\Users\park9\OneDrive\바탕 화면\Invariance_Driven_Distanglement\MHEALTHDATASET\mHealth_subject5.log"

SUBJECT_ID = 5
ACTIVITY_CODE = 8  # Knees Bending
FS = 50
OUTPUT_JSON = "./subject5_intraclass_gt.json"

# detector params (학습 코드와 동일하게 맞추기)
SCALE_FACTOR = 10.0
STABILITY_THRESHOLD = 0.5

# detector 내부 rolling window = 0.25s
ROLL_SEC = 0.25


# ==========================================
# Detector (same definition as your training)
# ==========================================
class InvertedInertialDetector:
    def __init__(self, fs=50, scale_factor=10.0, stability_threshold=0.5):
        self.fs = fs
        self.scale_factor = scale_factor
        self.stability_threshold = stability_threshold

    def detect(self, acc, gyro):
        """
        acc: (N,3), gyro: (N,3)
        return: stability_score (N,)
        """
        w_size = max(1, int(ROLL_SEC * self.fs))

        acc_df = pd.DataFrame(acc, columns=['x', 'y', 'z'])
        var_s = acc_df.rolling(window=w_size, center=True, min_periods=1).var().sum(axis=1).values
        var_s = np.nan_to_num(var_s)
        norm_var = np.clip(var_s / 5.0, 0, 1)

        gyro_mag = np.linalg.norm(gyro, axis=1)
        gyro_series = pd.Series(gyro_mag)
        smooth_gyro = gyro_series.rolling(window=w_size, center=True, min_periods=1).mean().values
        norm_gyro = np.clip(smooth_gyro / 300.0, 0, 1)

        total_energy = norm_var + norm_gyro
        stability_score = np.exp(-self.scale_factor * total_energy)
        return stability_score


# ==========================================
# Labeler
# ==========================================
class IntraClassLabeler:
    def __init__(self, data_path, activity_code, fs):
        self.fs = fs
        self.segments = []  # list of (start_sec, end_sec)

        print(f"Loading {data_path}...")
        try:
            df = pd.read_csv(data_path, sep=r"\s+", header=None, engine='python')
            data = df.values
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
            return
        except Exception as e:
            print(f"❌ 데이터 로드 중 에러 발생: {e}")
            return

        # mHealth: last column is activity label
        labels = data[:, -1]
        indices = np.where(labels == activity_code)[0]
        if len(indices) == 0:
            print(f"⚠️ Activity {activity_code}를 찾을 수 없습니다.")
            return

        self.start_idx = indices[0]
        self.end_idx   = indices[-1] + 1

        # ✅ training 코드와 동일 채널 사용 (ankle acc: 5,6,7 / ankle gyro: 8,9,10)
        X = data[self.start_idx:self.end_idx, 5:11].astype(np.float64)  # (N,6)
        self.raw = X.copy()

        # (권장) segment 단위 z-normalize: 모델/탐지 정의와 더 안정적으로 맞음
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        Xn = (X - mean) / std
        self.Xn = Xn

        acc = Xn[:, :3]
        gyro = Xn[:, 3:6]
        self.gyro_mag = np.linalg.norm(gyro, axis=1)

        # detector score
        self.det = InvertedInertialDetector(fs=fs, scale_factor=SCALE_FACTOR, stability_threshold=STABILITY_THRESHOLD)
        self.score = self.det.detect(acc, gyro)

        self.t = np.arange(len(Xn)) / fs

        # ----- Figure (2 subplots) -----
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        # SpanSelector는 아래(score) 축에 붙이기 (stable 정의와 직접 연결)
        self.span = SpanSelector(
            self.ax2, self.onselect, 'horizontal', useblit=True,
            props=dict(alpha=0.25, facecolor='green'),
            interactive=True, drag_from_anywhere=True
        )

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.plot()

        print("\n" + "="*60)
        print("🎮 [라벨링 가이드: 지금은 'STABLE 구간'을 라벨링합니다]")
        print(" - 아래 그래프(stability score)가 threshold(점선) 위로 올라간 구간을 드래그하세요.")
        print(" - 즉, '동작-동작 사이'에서 score가 높게 유지되는 구간(상대적으로 안정)을 선택.")
        print("\n[사용법]")
        print("1) 아래 score plot에서 마우스 드래그: stable 구간 추가")
        print("2) 키보드 'd': 최근 구간 삭제")
        print("3) 키보드 's': 저장(JSON)")
        print("4) 키보드 'q': 종료")
        print("="*60 + "\n")

    def start(self):
        plt.show()

    def plot(self):
        self.ax1.clear()
        self.ax2.clear()

        # --- Top: gyro magnitude ---
        self.ax1.plot(self.t, self.gyro_mag, color='black', linewidth=1.0, alpha=0.8, label='Gyro |w| (norm)')
        self.ax1.set_ylabel("Gyro |w|")
        self.ax1.grid(True, alpha=0.25)
        self.ax1.legend(loc='upper right')

        # --- Bottom: detector stability score ---
        self.ax2.plot(self.t, self.score, color='tab:blue', linewidth=1.5, label='Detector stability score')
        self.ax2.axhline(STABILITY_THRESHOLD, color='tab:red', linestyle='--', linewidth=1.2,
                         label=f"threshold={STABILITY_THRESHOLD:.2f}")

        # stable shading (score > thr)
        stable_mask = (self.score > STABILITY_THRESHOLD).astype(np.float32)
        self.ax2.fill_between(self.t, 0, 1, where=(stable_mask > 0.5),
                              alpha=0.08, color='green', transform=self.ax2.get_xaxis_transform(),
                              label='score > thr (hint)')

        self.ax2.set_ylim(-0.02, 1.02)
        self.ax2.set_ylabel("Stability (0~1)")
        self.ax2.set_xlabel("Time (sec)")
        self.ax2.grid(True, alpha=0.25)
        self.ax2.legend(loc='upper right')

        # --- Selected segments shading (both axes) ---
        for s, e in self.segments:
            self.ax1.axvspan(s, e, color='green', alpha=0.18)
            self.ax2.axvspan(s, e, color='green', alpha=0.18)

        self.fig.suptitle(f"Subj {SUBJECT_ID} - Act {ACTIVITY_CODE} (Manual STABLE Labeling)", fontsize=14)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.canvas.draw_idle()

    def onselect(self, xmin, xmax):
        if abs(xmax - xmin) < 0.05:
            return
        s = float(min(xmin, xmax))
        e = float(max(xmin, xmax))
        self.segments.append((s, e))
        print(f"✅ Added STABLE seg: {s:.2f} ~ {e:.2f} sec")
        self.plot()

    def save(self):
        out_data = {
            f"subject{SUBJECT_ID}_act{ACTIVITY_CODE}_stable": {
                "segments_sec": self.segments,
                "meta": {
                    "source": "Manual intra-class labeling (STABLE regions by detector definition)",
                    "fs": self.fs,
                    "activity_code": ACTIVITY_CODE,
                    "n_segments": len(self.segments),
                    "detector": {
                        "scale_factor": SCALE_FACTOR,
                        "stability_threshold": STABILITY_THRESHOLD,
                        "roll_sec": ROLL_SEC
                    }
                }
            }
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(out_data, f, indent=4)
        print(f"\n✅ Saved {len(self.segments)} stable segments to {OUTPUT_JSON}")

    def delete_last(self):
        if self.segments:
            removed = self.segments.pop()
            print(f"🗑️ Removed: {removed}")
            self.plot()
        else:
            print("⚠️ 삭제할 구간이 없습니다.")

    def on_key(self, event):
        if event.key == 's':
            self.save()
        elif event.key == 'd':
            self.delete_last()
        elif event.key == 'q':
            plt.close(self.fig)


if __name__ == "__main__":
    labeler = IntraClassLabeler(DATA_PATH, ACTIVITY_CODE, FS)
    if hasattr(labeler, 'fig'):
        labeler.start()
