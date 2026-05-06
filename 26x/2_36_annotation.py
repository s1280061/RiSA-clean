import os
import json
import time
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

# ==================================================
# CONFIG
# ==================================================
IMAGE_DIR = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted\raw"
QUESTION_CSV = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted\vlm_prompts_23_en.csv"
OUTPUT_JSON = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted\annotations_human.json"

ANNOTATOR_ID = "annotator_A"
IMAGE_SIZE = (960, 540)

# ==================================================
# LOAD DATA
# ==================================================
questions = pd.read_csv(QUESTION_CSV)

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

# ==================================================
# LOAD OR INIT JSON
# ==================================================
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {
        "annotator_id": ANNOTATOR_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotations": {}
    }

# ==================================================
# RESUME LOGIC
# ==================================================
def find_resume_position():
    """
    Find the first (question, image) pair that has not been annotated yet.
    """
    for q_idx, q in questions.iterrows():
        qid = f"Q{q.question_id}"

        answered_images = set()
        if qid in data["annotations"]:
            answered_images = set(data["annotations"][qid]["answers"].keys())

        for img_idx, img_name in enumerate(image_files):
            if img_name not in answered_images:
                return q_idx, img_idx

    return None, None  # all done


resume_q_idx, resume_img_idx = find_resume_position()

if resume_q_idx is None:
    print("All annotations are already completed.")
    exit()

# ==================================================
# STATE
# ==================================================
current_q_idx = resume_q_idx
current_img_idx = resume_img_idx
history = []  # stack for undo (q_idx, img_idx)

# ==================================================
# TKINTER UI
# ==================================================
root = tk.Tk()
root.title("Human Annotation Tool (By Question)")

img_label = tk.Label(root)
img_label.pack()

question_label = tk.Label(
    root,
    text="",
    wraplength=900,
    font=("Arial", 15)
)
question_label.pack(pady=10)

entry = tk.Entry(root, font=("Arial", 14), width=50)
entry.pack()
entry.focus()

hint_label = tk.Label(
    root,
    text="Enter: next | Ctrl+Z: undo | Type 'Not sure' if unsure",
    font=("Arial", 10)
)
hint_label.pack(pady=4)

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack()

# ==================================================
# FUNCTIONS
# ==================================================
def load_image():
    img_path = os.path.join(IMAGE_DIR, image_files[current_img_idx])
    img = Image.open(img_path)
    img = img.resize(IMAGE_SIZE)
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo


def load_question():
    q = questions.iloc[current_q_idx]
    qid = f"Q{q.question_id}"

    question_label.config(
        text=f"[{qid}] {q.question_text}"
    )

    status_label.config(
        text=f"{qid} | Image {current_img_idx + 1}/{len(image_files)}"
    )

    if qid not in data["annotations"]:
        data["annotations"][qid] = {
            "question_text": q.question_text,
            "answers": {}
        }


def save_json():
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_answer(answer):
    q = questions.iloc[current_q_idx]
    qid = f"Q{q.question_id}"
    img_name = image_files[current_img_idx]

    history.append((current_q_idx, current_img_idx))
    data["annotations"][qid]["answers"][img_name] = answer

    save_json()


def next_state():
    global current_img_idx, current_q_idx

    current_img_idx += 1
    if current_img_idx >= len(image_files):
        current_img_idx = 0
        current_q_idx += 1

        if current_q_idx >= len(questions):
            root.quit()
            return

    load_image()
    load_question()


def on_enter(event):
    answer = entry.get().strip()
    if answer == "":
        return

    save_answer(answer)
    entry.delete(0, tk.END)
    next_state()


def undo_last(event):
    global current_q_idx, current_img_idx

    if not history:
        return

    prev_q_idx, prev_img_idx = history.pop()

    q = questions.iloc[prev_q_idx]
    qid = f"Q{q.question_id}"
    img_name = image_files[prev_img_idx]

    if img_name in data["annotations"][qid]["answers"]:
        del data["annotations"][qid]["answers"][img_name]

    save_json()

    current_q_idx = prev_q_idx
    current_img_idx = prev_img_idx

    load_image()
    load_question()
    entry.delete(0, tk.END)

# ==================================================
# KEY BINDINGS
# ==================================================
root.bind("<Return>", on_enter)
root.bind("<Control-z>", undo_last)

# ==================================================
# START
# ==================================================
load_image()
load_question()
root.mainloop()
