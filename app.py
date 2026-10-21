import os
import tempfile
import shutil
import re
import pyrxing
import gradio as gr
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if gr.NO_RELOAD:
    MODEL = YOLO("qr_yolo26m_0.pt")

UPLOAD_DIR = tempfile.mkdtemp(prefix="grin_uploads_")

# bunch of custom CSS to adjust UI appearance
css = """
.vertical-radio .wrap { flex-direction: column; }
.vertical-radio [data-testid="block-info"] { font-size: 1.5rem; }
.label-wrap span { font-size: 1.8rem; }
.spaced { margin-top: 2.5rem; }
.large-label [data-testid="block-info"] { font-size: 1.5rem; }
.info-pane { padding: 1.5rem 1.25rem; }
.app-title h1 {
    font-family: "Montserrat", sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 3rem;
    color: #007680;
}
.padded-accordion { padding: 0.75rem; }
.vertical-radio { --block-info-text-color: #52525b; }
.vertical-radio [data-testid="block-info"] { color: #3f3f46; }
.centered-row { justify-content: center; }
#detect-progress .progress-bar-wrap { height: 1.5rem; }
#detect-progress .progress-level-inner { margin: 0.75rem auto; }
#detect-progress .meta-text,
#detect-progress .meta-text-center {
    position: static;
    transform: none;
    order: 1;
    margin-top: 0.5rem;
}
"""

BI_TEAL = "#007680"
BI_TEAL_HOVER = "#00646d"

bi_theme = gr.themes.Default(
    font=gr.themes.GoogleFont("Lato"),
    primary_hue="teal",
    text_size="lg",
    ).set(
    body_background_fill="#f4f6f9",
    button_primary_background_fill=BI_TEAL,
    button_primary_background_fill_hover=BI_TEAL_HOVER,
    button_primary_border_color=BI_TEAL,
    button_primary_border_color_hover=BI_TEAL_HOVER,
    button_primary_text_color="white",
    )
# Gradio has no light-only option, so copy every light-mode theme value onto
# its *_dark counterpart; dark mode then renders identically to light mode.
bi_theme.set(**{
    name: getattr(bi_theme, name.removesuffix("_dark"))
    for name in vars(bi_theme)
    if name.endswith("_dark")
})

# create a separator dictionary that we can reuse, instead of 
SEPARATORS = [{"string":"_",
               "label":"Underscore ( _ )"},
               {"string":"-",
               "label":"Hyphen ( - )"},
               {"string":" ",
               "label":"Space"},
               {"string":"",
                "label":"None"},
               {"string":"custom",
               "label":"Custom"}]

KEEP_EXTENSION = "Keep existing extensions"
EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
PIL_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}

TABLE_COLUMNS = ["Original filename", "QRs\ndetected", "QRs\nparsed", "Parsed text", "New filename"]

def toggle_custom(choice):
    return gr.Textbox(visible=(choice == "Custom"))


def toggle_custom_sep(index):
    return gr.Textbox(visible=(index is not None and SEPARATORS[index]["label"] == "Custom"))


def _resolve_affix(choice, custom_text, original_stem):
    """Return the prefix/suffix string for the selected Radio option."""
    if choice == "Custom":
        return custom_text or ""
    if choice == "Current filename":
        return original_stem
    return ""


def _resolve_separator(index, custom_text, default="_"):
    if index is None:
        return default
    entry = SEPARATORS[index]
    if entry["label"] == "Custom":
        return custom_text or ""
    return entry["string"]


def handle_upload(files):
    """Copy uploaded files to temp dir; return dataframe + first image preview."""
    empty_df = pd.DataFrame(columns=TABLE_COLUMNS)
    if not files:
        return empty_df, None, None, "### 0 image(s) uploaded", {}

    rows = []
    for f in sorted(files, key=lambda x: os.path.basename(x)):
        basename = os.path.basename(f)
        dest = os.path.join(UPLOAD_DIR, basename)
        shutil.copy2(f, dest)
        rows.append({"Original filename": basename, "QRs\ndetected": "", "QRs\nparsed": "",
                     "Parsed text": "", "New filename": ""})

    df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    first_image = Image.open(os.path.join(UPLOAD_DIR, df.iloc[0]["Original filename"]))
    count_text = f"### {len(df)} image(s) uploaded"
    return df, first_image, None, count_text, {}


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


def run_pipeline(df, prefix_choice, custom_prefix, split_index, custom_split, sep_index, custom_sep,
                 suffix_choice, custom_suffix, extension_choice, progress=gr.Progress(track_tqdm=False)):
    """Run YOLO detection on each uploaded image and count aztec_code hits."""
    if df is None or df.empty:
        gr.Info("No images to process. Upload images first.")
        return df, {}

    filenames = df["Original filename"].tolist()
    qr_counts = []
    parsed_qr_counts = []
    parsed_text = []
    for i, filename in progress.tqdm(enumerate(filenames), total=len(filenames), desc="Running detection"):
        path = os.path.join(UPLOAD_DIR, filename)
        img =  Image.open(path)
        r = MODEL(img, imgsz=640, conf=0.7, device=DEVICE, verbose=False)[0]
        count = 0
        parse_count = 0
        parser_results = []
        for box in r.boxes:
            if r.names[int(box.cls)] == "aztec_code":
                count += 1
                # try parsing the QR, 40px pad usually more than enough
                pad = 40
                x1,y1,x2,y2 = list(box.xyxy.cpu().numpy()[0])
                qr_crop = img.crop((x1-pad,y1-pad,x2+pad,y2+pad))
                rxing_result = pyrxing.read_barcode(qr_crop)
                if not rxing_result:
                    # retry on a 2x upscale in case code is small/blurry
                    qr_crop = qr_crop.resize((qr_crop.width * 2, qr_crop.height * 2), Image.LANCZOS)
                    rxing_result = pyrxing.read_barcode(qr_crop)
                if rxing_result and rxing_result.text:
                    parse_count += 1
                    parser_results.append(rxing_result.text)
        if len(parser_results) > 0:
            text = max(parser_results, key=len)
        else:
            text = ""
        qr_counts.append(count)
        parsed_qr_counts.append(parse_count)
        parsed_text.append(text)
    df["QRs\ndetected"] = qr_counts
    df["QRs\nparsed"] = parsed_qr_counts
    # Only show the parsed text when it's unambiguous (exactly one QR read).
    df["Parsed text"] = [t if n == 1 else "" for t, n in zip(parsed_text, parsed_qr_counts)]
    raw_qr = dict(zip(filenames, parsed_text))
    df = update_filenames(df, raw_qr, prefix_choice, custom_prefix, split_index, custom_split,
                          sep_index, custom_sep, suffix_choice, custom_suffix, extension_choice)
    gr.Info("Detection finished!")
    return df, raw_qr


def _format_name_from_qr(raw_text, splitchar, separator):
    """Split QR text on the user-selected split string and join with user-selected separator."""
    if raw_text is None:
        return ""
    text = str(raw_text).strip()
    if not text:
        return ""
    # An empty split string (the "None" option) means don't split at all.
    parts = [p for p in text.split(splitchar) if p] if splitchar else [text]
    joined = separator.join(parts).strip()
    # Keep names filesystem-safe.
    return re.sub(r"[\\/:*?\"<>|]+", "", joined).strip(" .")


def _build_new_name(raw_text, original, prefix_choice, custom_prefix, splitchar, sep, suffix_choice,
                    custom_suffix, extension_choice):
    body = _format_name_from_qr(raw_text, splitchar, sep)
    if not body:
        return ""
    stem, original_ext = os.path.splitext(original)
    name = (_resolve_affix(prefix_choice, custom_prefix, stem)
            + body
            + _resolve_affix(suffix_choice, custom_suffix, stem))
    name = re.sub(r"[\\/:*?\"<>|]+", "", name).strip(" .")
    ext = extension_choice if extension_choice in EXTENSIONS else original_ext
    return f"{name}{ext}"


def update_filenames(df, raw_qr, prefix_choice, custom_prefix, split_index, custom_split,
                     sep_index, custom_sep, suffix_choice, custom_suffix, extension_choice):
    """Rebuild the New filename column from the raw QR text and current settings."""
    if df is None or df.empty:
        return df
    raw_qr = raw_qr or {}
    splitchar = _resolve_separator(split_index, custom_split, default=" ")
    sep = _resolve_separator(sep_index, custom_sep)
    df["New filename"] = [
        _build_new_name(raw_qr.get(f, ""), f, prefix_choice, custom_prefix, splitchar, sep,
                        suffix_choice, custom_suffix, extension_choice)
        for f in df["Original filename"]
    ]
    return df


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


def download_renamed(df, include_mode, zip_name):
    if df is None or df.empty:
        gr.Warning("No files are available to download.")
        return None

    include_unparsed = include_mode == "Include images with no parsed QR"

    zip_stem = str(zip_name or "").strip()
    if zip_stem.lower().endswith(".zip"):
        zip_stem = zip_stem[:-4]
    zip_stem = re.sub(r"[\\/:*?\"<>|]+", "", zip_stem).strip(" .") or "renamed_images"

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

        ext = os.path.splitext(original)[1]
        new_name = re.sub(r"[\\/:*?\"<>|]+", "", str(new_name_raw)).strip(" .")

        if not new_name:
            if not include_unparsed:
                skipped += 1
                continue
            new_name = original
        elif os.path.splitext(new_name)[1] not in EXTENSIONS:
            # Append rather than replace, so dots in QR text aren't treated as extensions.
            new_name = f"{new_name}{ext}"

        final_name = _dedupe_filename(new_name, used_names)
        destination = os.path.join(renamed_dir, final_name)
        target_format = PIL_FORMATS.get(os.path.splitext(final_name)[1].lower())
        with Image.open(source) as img:
            if target_format is None or img.format == target_format:
                shutil.copy2(source, destination)
            else:
                save_kwargs = {}
                if "exif" in img.info:
                    save_kwargs["exif"] = img.info["exif"]
                if target_format == "JPEG":
                    img = img.convert("RGB")
                    save_kwargs["quality"] = 95
                img.save(destination, format=target_format, **save_kwargs)
        copied += 1

    if copied == 0:
        gr.Warning("No files matched your download settings.")
        return None

    zip_path = shutil.make_archive(os.path.join(run_dir, zip_stem), "zip", renamed_dir)
    gr.Info(f"Prepared {copied} file(s) for download ({skipped} skipped).")
    return gr.update(value=zip_path, visible=True)


with gr.Blocks(title="QR Image Renamer") as demo:
    raw_qr = gr.State({})
    with gr.Row():
        gr.Markdown("# Breeding Insight QR Image Renamer ", elem_classes="app-title")
    with gr.Row(elem_classes="spaced"): # Instructions + upload section?
        with gr.Column(scale=3):
            upload = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                show_label=False,
            )
            upload_count = gr.Markdown("### 0 image(s) uploaded")
            run_btn = gr.Button("Start Detection", variant="primary")
            # Empty placeholder that hosts the detection progress bar
            detect_progress = gr.Markdown("", min_height=120, elem_id="detect-progress")
        with gr.Column(scale=7):
            gr.Markdown( 
                """
                ### INSTRUCTIONS
                1. Upload your image(s) at left and hit **Start Detection** to begin the
                QR detection and reading process.
                2. Adjust your settings to tell the app how to construct file names .
                3. Use the filename table to make any manual corrections or additions to the
                final filenames. To view an image, click anywhere on its corresponding row.
                4. When ready, hit **Prepare Download** to generate a .zip of the renamed images,
                then hit **Download Renamed Images** to save them to your computer.

                ### INFO
                This app was created by [Breeding Insight](https://breedinginsight.org/),
                a USDA-funded initiative based at University of Florida. It is licensed under the open-source
                [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

                To run this app locally or on your own web server, or to fork the project to modify
                it to fit your own program's needs, please visit the app's
                [Github page](https://github.com/tyrwh/GRIN_image_renamer).

                **All data uploaded to the web version of this app is temporary.** Images and any associated data
                are saved in temporary storage and deleted after it has been dormant for one hour.
                
                For any questions or feedback,
                please email me at [twiesnerhanks@ufl.edu](mailto:twiesnerhanks@ufl.edu).

                Many thanks to Becky Povilus of the USDA-ARS for her help in creating and testing this app.
                
                """,
                container=True,
                elem_classes="info-pane",
            )

    with gr.Accordion(label="Filename settings", elem_classes=["spaced", "padded-accordion"]): # Settings dropdown
        with gr.Row():
            with gr.Column(scale=0, min_width=270):
                splitchar = gr.Radio(
                    label="Split QR text on...",
                    info="Character(s) to use when splitting the parsed text.",
                    choices=[x["label"] for x in SEPARATORS],
                    value=SEPARATORS[2]["label"],
                    type="index",
                    elem_classes="vertical-radio"
                    )
                custom_splitchar = gr.Textbox(visible=False,
                                        label="Custom separator to use:")
                splitchar.change(toggle_custom_sep, splitchar, custom_splitchar, show_progress="hidden")
            with gr.Column(scale=0, min_width=270):
                separator = gr.Radio(
                    label="Rejoin QR text on...",
                    info="Character(s) to use when rejoining the elements of the QR text.",
                    choices=[x["label"] for x in SEPARATORS],
                    value=SEPARATORS[0]["label"],
                    type="index",
                    elem_classes="vertical-radio"
                    )
                custom_sep = gr.Textbox(visible=False,
                                        label="Custom separator to use:")
                separator.change(toggle_custom_sep, separator, custom_sep, show_progress="hidden")
            with gr.Column(scale=0, min_width=270):
                prefix = gr.Radio(
                    choices=[
                        "None",
                        "Current filename",
                        "Custom",
                    ],
                    value="None",
                    label="Prefix",
                    info="Insert a prefix at the start of each filename.",
                    elem_classes="vertical-radio"
                )
                custom_prefix = gr.Textbox(visible=False,
                                        label="Custom prefix to use:")
                prefix.change(toggle_custom, prefix, custom_prefix, show_progress="hidden")

            with gr.Column(scale=0, min_width=270):
                suffix = gr.Radio(
                    choices=[
                        "None",
                        "Current filename",
                        "Custom",
                    ],
                    value="None",
                    label="Suffix",
                    info="Append a suffix to the end of each filename.",
                    elem_classes="vertical-radio"
                )
                custom_suffix = gr.Textbox(visible=False,
                                        label="Custom suffix to use:")
                suffix.change(toggle_custom, suffix, custom_suffix, show_progress="hidden")
            with gr.Column(scale=0, min_width=270):
                extension = gr.Radio(
                    choices=[KEEP_EXTENSION] + EXTENSIONS,
                    value=KEEP_EXTENSION,
                    label="Extension",
                    info="Filename extension to apply to all images.",
                    elem_classes="vertical-radio",
                )
        with gr.Row(elem_classes="centered-row"):
            update_btn = gr.Button(
                "Update filenames with current settings",
                variant="primary",
                scale=0,
                min_width=600,
                )
    with gr.Row(elem_classes="spaced"): # Results table
        table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            datatype=["str", "number", "number", "str", "str"],
            interactive=True,
            max_height=550,
            column_widths=["30%", "8%", "8%", "24%", "30%"],
            static_columns=[0, 1, 2, 3],
            row_count=(0, "fixed"),  # rows come only from uploads; hides the add-row controls
        )
    with gr.Row(): # Image viewer
        preview = gr.Image(label="Image preview", type="pil", height=500)
    with gr.Accordion(label="Download", elem_classes=["spaced", "padded-accordion"]): # Settings dropdown
        with gr.Row():
            with gr.Column(scale=0, min_width=400):
                include_unparsed = gr.Radio(
                    label="Unreadable QRs",
                    info="Should images with no detectable or readable QR code be included (with their original filename) in the final download?",
                    choices=[
                        "Include images with no parsed QR",
                        "Remove images with no parsed QR",
                    ],
                    value="Include images with no parsed QR",
                    elem_classes="vertical-radio",
                )
            with gr.Column():
                download_zip_name = gr.Textbox(
                    label="Output .zip name",
                    info="Filename to use for the final .zip file of renamed images.",
                    placeholder="renamed_images",
                    type="text",
                    elem_classes="large-label",
                )
            with gr.Column():
                prepare_btn = gr.Button("Prepare Download", variant="primary")
                download_btn = gr.DownloadButton("Download Renamed Images", variant="primary", visible=False)

    upload.upload(
        fn=handle_upload,
        inputs=[upload],
        outputs=[table, preview, upload, upload_count, raw_qr],
    )

    table.select(
        fn=preview_selected,
        inputs=[table],
        outputs=[preview],
    )

    naming_inputs = [prefix, custom_prefix, splitchar, custom_splitchar, separator, custom_sep, suffix, custom_suffix, extension]

    run_btn.click(
        fn=run_pipeline,
        inputs=[table] + naming_inputs,
        outputs=[table, raw_qr],
        show_progress_on=detect_progress,
    )

    update_btn.click(
        fn=update_filenames,
        inputs=[table, raw_qr] + naming_inputs,
        outputs=[table],
    )

    prepare_btn.click(
        fn=download_renamed,
        inputs=[table, include_unparsed, download_zip_name],
        outputs=[download_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=7860,
                css=css,
                head='<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">',
                theme=bi_theme)