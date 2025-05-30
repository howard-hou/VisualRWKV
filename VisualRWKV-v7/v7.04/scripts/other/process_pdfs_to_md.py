import os
import shutil
import fitz  # PyMuPDF
import argparse
import subprocess
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def split_pdf_to_pages(input_pdf, output_dir):
    """Split a PDF into single-page PDFs with zero-padded filenames."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
    doc = fitz.open(input_pdf)
    num_pages = len(doc)
    num_digits = len(str(num_pages))

    for i in range(num_pages):
        padded_index = str(i).zfill(num_digits)
        output_path = os.path.join(output_dir, f"{pdf_name}-{padded_index}.pdf")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        new_doc.save(output_path)
        new_doc.close()

    doc.close()


def run_magic_pdf(input_dir, output_dir, env=None):
    """Run the external 'magic-pdf' command with the specified environment."""
    cmd = ["magic-pdf", "-p", input_dir, "-o", output_dir]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        return True
    except subprocess.CalledProcessError as e:
        return False


def move_md_files(temp_output_dir, final_output_dir, pdf_name):
    """Recursively move all .md files from temp_output_dir to final_output_dir/pdf_name."""
    target_dir = os.path.join(final_output_dir, pdf_name)
    os.makedirs(target_dir, exist_ok=True)

    for root, _, files in os.walk(temp_output_dir):
        for file in files:
            if file.endswith(".md"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.move(src_path, dst_path)


def process_pdf(args):
    """Process a single PDF file using a specified CUDA device."""
    input_pdf, output_dir, device = args
    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]

    with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        split_pdf_to_pages(input_pdf, temp_input_dir)
        # Set CUDA device for subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        success = run_magic_pdf(temp_input_dir, temp_output_dir, env)
        if not success:
            print(f"[ERROR] magic-pdf failed for: {pdf_name}")
            return None
        move_md_files(temp_output_dir, output_dir, pdf_name)
    return pdf_name


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using magic-pdf.")
    parser.add_argument("input_dir", help="Directory containing input PDF files")
    parser.add_argument("output_dir", help="Directory to store Markdown output")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--devices", type=str, default="0",
                    help="Comma-separated list of CUDA device IDs (e.g., 0,1,2)")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.max_workers
    devices = args.devices.split(",")
    num_devices = len(devices)

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")]
    pdf_files = sorted(pdf_files)
    tasks = [(pdf_path, output_dir, devices[i % num_devices]) for i, pdf_path in enumerate(pdf_files)]

    num_workers = min(max_workers, len(tasks))
    if num_workers <= 1:
        print("Running in single-process mode.")
        for task in tqdm(tasks, desc="Processing PDFs"):
            process_pdf(task)
        return
        
    print(f"Running with {num_workers} parallel workers.")
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_pdf, tasks), total=len(tasks), desc="Processing PDFs"):
            pass


if __name__ == "__main__":
    main()
