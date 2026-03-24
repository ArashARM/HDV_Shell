import os
import shutil   
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap


class TimelapseRecorder:
    def __init__(self, out_dir="timelapse_frames", video_path="timelapse.mp4", fps=10):
        self.out_dir = out_dir
        self.video_path = video_path
        self.fps = fps
        os.makedirs(out_dir, exist_ok=True)
        self.frame_paths = []

    def _make_loss_chart(self, loss_dict, title_text="", height=700, width=700):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111)

        keys = list(loss_dict.keys())
        vals = [float(loss_dict[k]) for k in keys]

        ax.bar(keys, vals)
        ax.set_title("Loss values")
        ax.tick_params(axis="x", rotation=45)

        ymax = max(vals) if len(vals) else 1.0
        ymax = max(ymax * 1.2, 1e-8)
        ax.set_ylim(0.0, ymax)

        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)

        max_len = len(title_text)
        fontsize = max(10, min(14, 600 // max_len * 10))
        wrapped_title = "\n".join(textwrap.wrap(title_text, width=50))
        fig.suptitle(wrapped_title, fontsize=12)

        fig.tight_layout(rect=[0, 0, 1, 0.9])


        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3].copy()   # RGB
        plt.close(fig)
        return img

    def add_frame(self, step, cad_img, loss_dict, title_text=""):
        if cad_img is None:
            raise ValueError("cad_img is None")

        if cad_img.ndim != 3 or cad_img.shape[2] not in (3, 4):
            raise ValueError(f"cad_img must be HxWx3 or HxWx4, got shape {cad_img.shape}")

        cad_img = cad_img[..., :3]

        h_left, w_left = cad_img.shape[:2]
        chart_img = self._make_loss_chart(
            loss_dict=loss_dict,
            title_text=f"iter {step} | {title_text}",
            height=h_left,
            width=max(600, int(0.75 * w_left)),
        )

        h_right, w_right = chart_img.shape[:2]

        target_h = max(h_left, h_right)

        if h_left != target_h:
            cad_img = cv2.resize(cad_img, (w_left, target_h))
        if h_right != target_h:
            chart_img = cv2.resize(chart_img, (w_right, target_h))

        # matplotlib gives RGB, cv2 prefers BGR when writing
        chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

        combined = np.hstack([cad_img, chart_img])

        frame_path = os.path.join(self.out_dir, f"frame_{step:06d}.png")
        cv2.imwrite(frame_path, combined)
        self.frame_paths.append(frame_path)

    def build_video(self,delete_frames=True):
        if not self.frame_paths:
            raise RuntimeError("No frames recorded.")

        first = cv2.imread(self.frame_paths[0])
        if first is None:
            raise RuntimeError(f"Could not read first frame: {self.frame_paths[0]}")

        h, w = first.shape[:2]

        writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )

        for fp in self.frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)

        writer.release()
        print(f"Saved video to: {self.video_path}")
        if delete_frames:
            try:
                shutil.rmtree(self.out_dir)
                print(f"Deleted frames directory: {self.out_dir}")
            except Exception as e:
                print(f"Warning: could not delete frames directory: {e}")