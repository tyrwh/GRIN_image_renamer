import os
import tempfile
import shutil
import time
import re
import zxing
import gradio as gr
import pandas as pd
from PIL import Image
from ultralytics import YOLO

MODEL = YOLO("qr_yolo26m_0.pt")
reader = zxing.BarCodeReader()

UPLOAD_DIR = tempfile.mkdtemp(prefix="grin_uploads_")

def handle_upload(files):
    """Copy uploaded files to temp dir; return dataframe + first image preview."""
    empty_df = pd.DataFrame(columns=["Original filename", "QRs\ndetected", "QRs\nparsed", "New filename"])
    if not files:
        return empty_df, None, None, "### 0 image(s) uploaded"

    rows = []
    for f in sorted(files, key=lambda x: os.path.basename(x)):
        basename = os.path.basename(f)
        dest = os.path.join(UPLOAD_DIR, basename)
        shutil.copy2(f, dest)
        rows.append({"Original filename": basename, "QRs\ndetected": "", "QRs\nparsed": "", "New filename": ""})

    df = pd.DataFrame(rows)
    first_image = Image.open(os.path.join(UPLOAD_DIR, df.iloc[0]["Original filename"]))
    count_text = f"### {len(df)} image(s) uploaded"
    return df, first_image, None, count_text


def preview_selected(evt: gr.SelectData, df):
    """Show the image for the clicked row."""
    if df is None or df.empty:
        return None
    row_idx = evt.index[0]
    filename = df.iloc[row_idx]["Original filename"]
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

def run_pipeline(df, progress=gr.Progress(track_tqdm=False)):
    """Run YOLO detection on each uploaded image and count aztec_code hits."""
    if df is None or df.empty:
        gr.Info("No images to process. Upload images first.")
        return df

    filenames = df["Original filename"].tolist()
    qr_counts = []
    parsed_qr_counts = []
    parsed_text = []
    for i, filename in progress.tqdm(enumerate(filenames), total=len(filenames), desc="Running detection"):
        path = os.path.join(UPLOAD_DIR, filename)
        img =  Image.open(path)
        r = MODEL(img, imgsz=640, conf=0.7, verbose=False)[0]
        count = 0
        parse_count = 0
        parser_results = []
        for box in r.boxes:
            if r.names[int(box.cls)] == "aztec_code":
                count += 1
                # try parsing the QR
                pad = 10
                x1,y1,x2,y2 = list(box.xyxy.numpy()[0])
                qr_crop = img.crop((x1-pad,y1-pad,x2+pad,y2+pad))
                zxing_results = reader.decode(qr_crop)
                if zxing_results and zxing_results.parsed:
                    parse_count += 1
                    parser_results.append(zxing_results.parsed)
        if len(parser_results) > 0:
            text = max(parser_results, key=len)
        else:
            text = ""
        qr_counts.append(count)
        parsed_qr_counts.append(parse_count)
        parsed_text.append(text)
    df["QRs\ndetected"] = qr_counts
    df["QRs\nparsed"] = parsed_qr_counts
    df["New filename"] = parsed_text
    gr.Info("Pipeline finished!")
    return df

def _format_name_from_qr(raw_text, separator):
    """Split QR text on spaces and join with user-selected separator."""
    if raw_text is None:
        return ""
    text = str(raw_text).strip()
    if not text:
        return ""
    parts = [p for p in text.split(" ") if p]
    joined = separator.join(parts).strip()
    # Keep names filesystem-safe.
    return re.sub(r"[\\/:*?\"<>|]+", "", joined).strip(" .")


def _dedupe_filename(filename, used_names):
    """Ensure duplicate names are made unique with a numeric suffix."""
    stem, ext = os.path.splitext(filename)
    candidate = filename
    counter = 2
    while candidate in used_names:
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    used_names.add(candidate)
    return candidate


def download_renamed(df, separator, include_mode):
    if df is None or df.empty:
        gr.Warning("No files are available to download.")
        return None

    sep = separator if separator not in (None, "") else "_"
    include_unparsed = include_mode == "Include images with no parsed QR"

    run_dir = tempfile.mkdtemp(prefix="grin_renamed_")
    renamed_dir = os.path.join(run_dir, "renamed_images")
    os.makedirs(renamed_dir, exist_ok=True)

    copied = 0
    skipped = 0
    used_names = set()

    for _, row in df.iterrows():
        original = str(row.get("Original filename", "")).strip()
        if not original:
            continue

        source = os.path.join(UPLOAD_DIR, original)
        if not os.path.exists(source):
            continue

        new_name_raw = row.get("New filename", "")
        if pd.isna(new_name_raw):
            new_name_raw = ""

        stem, ext = os.path.splitext(original)
        renamed_stem = _format_name_from_qr(new_name_raw, sep)

        if not renamed_stem:
            if not include_unparsed:
                skipped += 1
                continue
            renamed_stem = stem

        final_name = _dedupe_filename(f"{renamed_stem}{ext}", used_names)
        destination = os.path.join(renamed_dir, final_name)
        shutil.copy2(source, destination)
        copied += 1

    if copied == 0:
        gr.Warning("No files matched your download settings.")
        return None

    zip_path = shutil.make_archive(os.path.join(run_dir, "renamed_images"), "zip", renamed_dir)
    gr.Info(f"Prepared {copied} file(s) for download ({skipped} skipped).")
    return gr.update(value=zip_path, visible=True)


with gr.Blocks(title="GRIN Image Renamer") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
                # GRIN Image Renamer

                This is a minimal app to rename images using information scraped from
                QR / Aztec codes detected in the image.
                
                Simply select images to upload, then click **Run Pipeline** to process them.
                After the pipeline has run, you can inspect the new image names and make any 
                necessary changes before downloading the renamed versions.

                *Note*: If you are running this app on your local computer via Docker Desktop,
                then the images are not uploaded to a web server of any kind. They are simply
                "uploaded" to a temporary storage location on your machine until the app is shut
                down.

                This app was written by Tyr Wiesner-Hanks of Breeding Insight, a USDA-funded
                initiative based at University of Florida. If you have any questions or feedback,
                please email me at [twiesnerhanks@ufl.edu](mailto:twiesnerhanks@ufl.edu).                
                """,
                container=True
            )
        with gr.Column(scale=2):
            upload = gr.File(
                label="Upload images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            upload_count = gr.Markdown("### 0 image(s) uploaded")
            run_btn = gr.Button("Run Pipeline", variant="primary")
        with gr.Column(scale=3):
            gr.Markdown("### Download settings")
            separator = gr.Textbox(label="Separator", value="_", max_lines=1)
            include_unparsed = gr.Radio(
                choices=[
                    "Include images with no parsed QR",
                    "Exclude images with no parsed QR",
                ],
                value="Include images with no parsed QR",
                label="Unparsed images",
            )
            prepare_btn = gr.Button("Prepare Download", variant="secondary")
            download_btn = gr.DownloadButton("Download renamed images", visible=False)

    with gr.Row(max_height=600):
        with gr.Column(scale=1):
            table = gr.Dataframe(
                headers=["Original filename", "QRs\ndetected", "QRs\nparsed", "New filename"],
                datatype=["str", "number", "number", "str"],
                interactive=True,
                max_height=550,
                column_widths=["40%","10%","10%","40%"]
            )
        with gr.Column(scale=1):
            preview = gr.Image(label="Image preview", type="pil")

    # --- wiring ---
    upload.upload(
        fn=handle_upload,
        inputs=[upload],
        outputs=[table, preview, upload, upload_count],
    )

    table.select(
        fn=preview_selected,
        inputs=[table],
        outputs=[preview],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[table],
        outputs=[table],
    )

    prepare_btn.click(
        fn=download_renamed,
        inputs=[table, separator, include_unparsed],
        outputs=[download_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
