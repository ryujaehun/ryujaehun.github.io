import argparse
import requests
import re
import os
import hashlib
from datetime import datetime
import urllib3


def generate_log_filename(url):
    """
    Generate a consistent log filename based on the URL and current date.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"log_{date_str}_{url_hash}.txt"


def extract_pdf_links(url):
    """
    Extract PDF links from the given URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        pdf_links = re.findall(r"https?://[^\s]+\.pdf", content)
        return pdf_links
    else:
        print(f"Failed to fetch the URL: {url}")
        return []


def download_pdf(pdf_url, download_dir):
    """
    Download a PDF from a URL to the specified directory.
    Handles SSL verification issues gracefully.
    """
    try:
        response = requests.get(pdf_url, stream=True, verify=True)  # SSL 검증 활성화
    except requests.exceptions.SSLError as e:
        print(f"SSL Error encountered for {pdf_url}: {e}")
        print("Retrying with SSL verification disabled...")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            response = requests.get(pdf_url, stream=True, verify=False)  # SSL 검증 비활성화로 재시도
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {pdf_url} even after disabling SSL verification: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {pdf_url}: {e}")
        return None

    if response.status_code == 200:
        filename = os.path.join(download_dir, pdf_url.split("/")[-1])
        if os.path.exists(filename):
            print(f"File already exists, skipping: {filename}")
            return None  # Skip downloading if file already exists
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return filename
    else:
        print(f"Failed to download PDF: {pdf_url}, HTTP status code: {response.status_code}")
        return None


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Download PDF files from a GitHub README file.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The URL of the GitHub README file containing PDF links.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where downloaded PDFs will be saved.",
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate log file name
    log_file = generate_log_filename(args.url)

    # Extract PDF links
    pdf_links = extract_pdf_links(args.url)
    if not pdf_links:
        print("No PDF links found.")
        return

    # Load previously downloaded files from the log
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logged_files = set(line.split(",")[1].strip() for line in f if line.strip())
    else:
        logged_files = set()

    # Download PDFs and log the filenames
    with open(log_file, "a") as log:
        for pdf_url in pdf_links:
            if pdf_url in logged_files:
                print(f"URL already logged, skipping: {pdf_url}")
                continue

            print(f"Downloading: {pdf_url}")
            downloaded_file = download_pdf(pdf_url, args.output_dir)
            if downloaded_file:
                # Log the download
                log_entry = f"{datetime.now().isoformat()},{pdf_url},{downloaded_file}\n"
                log.write(log_entry)
                logged_files.add(pdf_url)

    print(f"Log file updated: {log_file}")


if __name__ == "__main__":
    main()