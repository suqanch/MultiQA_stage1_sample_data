import os
import subprocess

BASE_DIR = "sample_data"
JAR_PATH = "pdffigures2.jar" 
JAVA_OPTS = "-Xmx4g -Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider"

def run_pdffigures(pdf_path, out_dir):
    """ parse pdf """
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, "stats.json")
    fig_prefix = os.path.join(out_dir, "fig")
    data_prefix = os.path.join(out_dir, "data")

    cmd = [
        "java"
    ] + JAVA_OPTS.split() + [
        "-cp", JAR_PATH,
        "org.allenai.pdffigures2.FigureExtractorBatchCli",
        pdf_path,
        "-m", fig_prefix,
        "-s", stats_path,
        "-d", data_prefix,
        "--dpi", "300"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for folder in os.listdir(BASE_DIR):
        subdir = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(subdir):
            continue

        pdfs = [f for f in os.listdir(subdir) if f.lower().endswith(".pdf")]
        if len(pdfs) != 1:
            print(f"Skipping {subdir}, found {len(pdfs)} PDFs")
            continue
        
        pdf_path = os.path.join(subdir, pdfs[0])
        out_dir = os.path.join(subdir, "parse_result")
        print(f"Processing {subdir}, found PDF: {pdfs[0]}, pdf_path: {pdf_path}, out_dir: {out_dir}")
        run_pdffigures(pdf_path, out_dir)

if __name__ == "__main__":
    main()






# java -Xmx4g -cp pdffigures2.jar \
#   org.allenai.pdffigures2.FigureExtractorBatchCli \
#   sample_data/Chulo/Chulo.pdf \
#   -s sample_data/Chulo/parse_result/stats.json \
#   -d sample_data/Chulo/parse_result/data \
#   --ignore-error \
#   --dpi 300