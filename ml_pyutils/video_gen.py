import subprocess
from pathlib import Path

import cv2
import numpy as np
import pexpect


def run_ffmpeg_cmd(ffmpeg_cmd: str, cmd_timeout: int = 30 * 60) -> None:
    """
    Runs an arbitrary FFMPEG command in shell. Also extracts ffmpeg command progress in
    number of frames inf `update_progress` is True.
    """
    if not ffmpeg_cmd.startswith("ffmpeg"):
        msg = "Given command string is not a FFMPEG command."
        raise ValueError(msg)

    ffmpeg_proc = pexpect.spawn(ffmpeg_cmd, timeout=cmd_timeout)
    compile_patterns = ffmpeg_proc.compile_pattern_list(
        [pexpect.EOF, r"frame=\s*(\d+)"]
    )
    while True:
        i = ffmpeg_proc.expect_list(compile_patterns)

        if i == 0:  # EOF
            break

    ffmpeg_proc.wait()

    if ffmpeg_proc.exitstatus is not None and ffmpeg_proc.exitstatus != 0:
        e = subprocess.CalledProcessError(ffmpeg_proc.exitstatus, ffmpeg_cmd)
        error_str = ffmpeg_proc.before.decode("utf-8").strip()
        print(f"Error: {error_str}")
        raise e


def create_word_frames(
    width: int,
    height: int,
    num_channels: int,
    num_frames: int,
) -> np.ndarray
    frames = np.zeros((num_frames, height, width, num_channels), dtype=np.uint8)

    num_digits = len(str(num_frames - 1))
    hue = 0
    for i in range(num_frames):
        if num_frames != 1:
            hue = (i * 160) / (num_frames - 1)
        hsv_color = np.full((height, width, 3), (hue, 255, 255), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)

        frame_text = f"frame_{str(i).zfill(num_digits)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = width / 100
        font_color = (0, 0, 0)
        thickness = 0
        text_size = cv2.getTextSize(frame_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Put the text on the image
        cv2.putText(
            frame_rgb,
            frame_text,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            thickness,
        )

        # Save the frame to the frames array
        frames[i] = frame_rgb

    return frames


def create_random_frames(
    width: int,
    height: int,
    num_channels: int,
    num_frames: int,
) -> np.ndarray:
    frames = np.random.randint(0, 256, (num_frames, height, width, num_channels), dtype="uint8")
    return frames


def create_video_from_frames(
    path: str | Path,
    frames: np.ndarray,
    fps: float | int,
    codec: str = "FFV1",
):
    """
    Creates a video with random noise in it.
    """

    height, width, num_channels = frames[0].shape
    is_color = num_channels == 3
    out = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
        is_color,
    )
    for data in frames:
        out.write(data)

    out.release()


def create_test_video(
    path: str | Path,
    width: int,
    height: int,
    duration: int,
    fps: float | int,
    codec: str = "mp4v",
):
    """
    Creates a video with random noise in it.
    """
    out = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
        False,
    )
    for _ in range(int(fps * duration)):
        data = np.random.randint(0, 256, (height, width), dtype="uint8")
        out.write(data)

    out.release()
