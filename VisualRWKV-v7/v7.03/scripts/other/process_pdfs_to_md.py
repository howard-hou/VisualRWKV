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


def run_magic_pdf(input_dir, output_dir):
    """Run the external 'magic-pdf' command."""
    cmd = ["magic-pdf", "-p", input_dir, "-o", output_dir]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    """Process a single PDF file: split, run magic-pdf, move output."""
    input_pdf, output_dir = args
    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]

    with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        split_pdf_to_pages(input_pdf, temp_input_dir)
        run_magic_pdf(temp_input_dir, temp_output_dir)
        move_md_files(temp_output_dir, output_dir, pdf_name)
    return pdf_name


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using magic-pdf.")
    parser.add_argument("input_dir", help="Directory containing input PDF files")
    parser.add_argument("output_dir", help="Directory to store Markdown output")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers (default: 1)")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.max_workers or cpu_count()

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")]
    tasks = [(pdf_path, output_dir) for pdf_path in pdf_files]

    num_workers = min(max_workers, len(tasks))
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_pdf, tasks), total=len(tasks), desc="Processing PDFs"):
            pass


if __name__ == "__main__":
    main()
