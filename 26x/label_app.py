# label_app.py
# Simple lane-change labeling app (Streamlit)
#
# Features:
# - Load images from a folder (jpg/png/jpeg)
# - Show one image at a time
# - Label with buttons (Keep / LC Left / LC Right / Unclear)
# - Save to CSV incrementally (resume supported)
# - Next/Prev navigation

import os
import glob
import time
from datetime import datetime

import pandas as pd
from PIL import Image
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DEFAULT_LABELS = [
    ("keep_lane", "Keep lane"),
    ("lane_change_left", "Lane change LEFT"),
    ("lane_change_right", "Lane change RIGHT"),
    ("unclear", "Unclear / can't tell"),
]

SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def list_images(folder: str):
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    # Sort by filename for stable order
    paths = sorted(paths)
    return paths


def load_existing_labels(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # expected columns: image_path,label,labeled_at
            if "image_path" in df.columns and "label" in df.columns:
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["image_path", "label", "labeled_at"])


def upsert_label(df: pd.DataFrame, image_path: str, label: str) -> pd.DataFrame:
    now = datetime.now().isoformat(timespec="seconds")
    if (df["image_path"] == image_path).any():
        df.loc[df["image_path"] == image_path, ["label", "labeled_at"]] = [label, now]
    else:
        df = pd.concat(
            [df, pd.DataFrame([{"image_path": image_path, "label": label, "labeled_at": now}])],
            ignore_index=True,
        )
    return df


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Lane-change Labeler", layout="wide")
st.title("Lane-change Labeler (簡易ラベリングアプリ)")

with st.sidebar:
    st.header("Settings")

    folder = st.text_input(
        "画像フォルダ（例）",
        value=r"C:\Users\s1280\Desktop\eval_images",
        help="このフォルダ配下の jpg/png を読み込みます。",
    )

    output_csv = st.text_input(
        "出力CSVパス",
        value=r"C:\Users\s1280\Desktop\eval_images\lanechange_labels.csv",
        help="ラベルはここに追記/更新保存します（再開対応）。",
    )

    show_filename = st.checkbox("ファイル名を表示", value=True)
    auto_skip_labeled = st.checkbox("ラベル済みは自動でスキップ", value=True)

    st.markdown("---")
    st.caption("操作: ボタンでラベル → 自動で次へ（ラベル済みスキップONなら未ラベルへ飛びます）")


if not folder or not os.path.isdir(folder):
    st.warning("左のサイドバーで、存在する画像フォルダを指定してください。")
    st.stop()

image_paths = list_images(folder)
if len(image_paths) == 0:
    st.error("指定フォルダに画像が見つかりませんでした（jpg/png/webp）。")
    st.stop()

df = load_existing_labels(output_csv)
labeled_map = {row["image_path"]: row["label"] for _, row in df.iterrows()}

# session state
if "idx" not in st.session_state:
    st.session_state.idx = 0

def find_next_unlabeled(start_idx: int):
    n = len(image_paths)
    for k in range(n):
        i = (start_idx + k) % n
        if image_paths[i] not in labeled_map:
            return i
    return start_idx  # all labeled

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    if auto_skip_labeled:
        st.session_state.idx = find_next_unlabeled(st.session_state.idx)

idx = st.session_state.idx
current_path = image_paths[idx]

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    if show_filename:
        st.subheader(os.path.basename(current_path))
        st.caption(current_path)

    # Load and show image
    try:
        img = Image.open(current_path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"画像を開けませんでした: {e}")

with col_right:
    st.subheader("Label")

    existing = labeled_map.get(current_path, None)
    if existing is not None:
        st.info(f"現在のラベル: **{existing}**")

    st.markdown("### クリックでラベル付け")

    # Label buttons
    for label_value, label_text in DEFAULT_LABELS:
        if st.button(label_text, use_container_width=True):
            df2 = upsert_label(df, current_path, label_value)
            # Save immediately
            os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
            df2.to_csv(output_csv, index=False, encoding="utf-8-sig")

            # Update in-memory map
            labeled_map[current_path] = label_value

            # Move index forward (skip labeled if enabled)
            next_idx = (idx + 1) % len(image_paths)
            if auto_skip_labeled:
                next_idx = find_next_unlabeled(next_idx)
            st.session_state.idx = next_idx

            st.success(f"Saved: {label_value}")
            time.sleep(0.2)
            st.rerun()

    st.markdown("---")

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("⬅ Prev", use_container_width=True):
            st.session_state.idx = (idx - 1) % len(image_paths)
            st.rerun()
    with nav2:
        if st.button("⏭ Next", use_container_width=True):
            st.session_state.idx = (idx + 1) % len(image_paths)
            if auto_skip_labeled:
                st.session_state.idx = find_next_unlabeled(st.session_state.idx)
            st.rerun()
    with nav3:
        if st.button("⏩ Next Unlabeled", use_container_width=True):
            st.session_state.idx = find_next_unlabeled((idx + 1) % len(image_paths))
            st.rerun()

    st.markdown("---")
    total = len(image_paths)
    labeled_count = sum(1 for p in image_paths if p in labeled_map)
    st.metric("Progress", f"{labeled_count}/{total}")

    if st.button("💾 CSVを再読み込み", use_container_width=True):
        st.rerun()
