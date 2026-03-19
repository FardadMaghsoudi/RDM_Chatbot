#!/usr/bin/env python3
"""
Test script to verify PDF downloading, text extraction, and pickle saving.
"""

import os
import pickle
import tempfile
import shutil
from pdf_utils import download_pdfs_from_webpage, load_all_pdfs, save_or_load_pdf_chunks

def test_pdf_download_and_processing():
    print("=" * 80)
    print("Testing PDF Download and Processing")
    print("=" * 80)
    
    # Test URL - using TU Delft research data management page
    test_url = "https://www.tudelft.nl/en/library/data-management/research-data-management/tu-delft-faculty-policies-for-research-data"
    download_folder = "./test_policies"
    
    # Step 1: Download PDFs
    print("\n1. Downloading PDFs from webpage...")
    try:
        result_folder = download_pdfs_from_webpage(test_url, download_folder)
        print(f"✓ PDFs downloaded to: {result_folder}")
        
        # List downloaded files
        if os.path.exists(download_folder):
            files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
            print(f"✓ Found {len(files)} PDF file(s):")
            for f in files:
                file_path = os.path.join(download_folder, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size:,} bytes)")
        else:
            print("✗ Download folder not created")
            return
    except Exception as e:
        print(f"✗ Error downloading PDFs: {e}")
        return
    
    # Step 2: Load and extract text from PDFs (with individual file testing)
    print("\n2. Extracting text from PDFs...")
    
    # First, test each PDF individually to identify problematic files
    print("\n2a. Testing each PDF individually...")
    from pdf_utils import load_pdf_text
    import PyPDF2
    
    if os.path.exists(download_folder):
        pdf_files = [f for f in os.listdir(download_folder) if f.endswith('.pdf')]
        successful_pdfs = []
        failed_pdfs = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(download_folder, pdf_file)
            try:
                text = load_pdf_text(pdf_path)
                print(f"  ✓ {pdf_file}: {len(text):,} characters extracted")
                successful_pdfs.append((pdf_file, text))
            except PyPDF2.errors.PdfReadError as e:
                print(f"  ✗ {pdf_file}: PDF Read Error - {e}")
                failed_pdfs.append((pdf_file, str(e)))
            except Exception as e:
                print(f"  ✗ {pdf_file}: {type(e).__name__} - {e}")
                failed_pdfs.append((pdf_file, str(e)))
        
        if failed_pdfs:
            print(f"\n⚠ {len(failed_pdfs)} PDF(s) failed to process:")
            for pdf_file, error in failed_pdfs:
                print(f"  - {pdf_file}: {error}")
                # Try to identify if file is corrupted
                pdf_path = os.path.join(download_folder, pdf_file)
                file_size = os.path.getsize(pdf_path)
                if file_size < 1000:
                    print(f"    → File appears too small ({file_size} bytes), likely incomplete download")
        
        if not successful_pdfs:
            print("\n✗ No PDFs were successfully processed!")
            return
        
        # Combine text from successful PDFs
        print(f"\n2b. Combining text from {len(successful_pdfs)} successful PDF(s)...")
        pdf_text = "\n".join([text for _, text in successful_pdfs])
        text_length = len(pdf_text)
        lines = pdf_text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        print(f"✓ Extracted text:")
        print(f"  - Total characters: {text_length:,}")
        print(f"  - Total lines: {len(lines):,}")
        print(f"  - Non-empty lines: {len(non_empty_lines):,}")
        print(f"  - First 500 characters:")
        print("-" * 80)
        print(pdf_text[:500])
        print("-" * 80)
    else:
        print("✗ Download folder not found")
        return
    
    # Step 3: Test pickle saving and loading
    print("\n3. Testing pickle save/load functionality...")
    pickle_path = "./test_pdf_text.pkl"
    
    try:
        # Save to pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(pdf_text, f)
        pickle_size = os.path.getsize(pickle_path)
        print(f"✓ Saved to pickle: {pickle_path} ({pickle_size:,} bytes)")
        
        # Load from pickle
        with open(pickle_path, "rb") as f:
            loaded_text = pickle.load(f)
        
        # Verify integrity
        if loaded_text == pdf_text:
            print(f"✓ Pickle load successful - text matches exactly")
            print(f"  - Loaded {len(loaded_text):,} characters")
        else:
            print(f"✗ Pickle load failed - text mismatch!")
            print(f"  - Original: {len(pdf_text):,} chars")
            print(f"  - Loaded: {len(loaded_text):,} chars")
    except Exception as e:
        print(f"✗ Error with pickle operations: {e}")
        return
    
    
    
    print("\n" + "=" * 80)
    print("All tests completed successfully! ✓")
    print("=" * 80)
    
    # Cleanup info
    print("\nTest artifacts created:")
    print(f"  - {download_folder}/ (downloaded PDFs)")
    print(f"  - {pickle_path} (pickle test)")
    print("\nTo clean up test files, run:")
    print(f"  rm -rf {download_folder} {pickle_path}")


def simple_split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    if not chunks:
        chunks.append("")
    return chunks


def test_save_or_load_pdf_chunks():
    print("=" * 80)
    print("Testing save_or_load_pdf_chunks (no download required)")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_test_")
    pdf_chunks_path = os.path.join(temp_dir, "pdf_chunks.pkl")

    try:
        # First call should create the pickle after processing the (empty) folder
        print("\n1. First call - creating pickle from empty folder...")
        chunks_first = save_or_load_pdf_chunks(pdf_chunks_path, temp_dir, simple_split_text)
        print(f"  ✓ Returned {len(chunks_first)} chunk(s)")

        if os.path.exists(pdf_chunks_path):
            size_bytes = os.path.getsize(pdf_chunks_path)
            print(f"  ✓ Pickle created at {pdf_chunks_path} ({size_bytes} bytes)")
        else:
            print("  ✗ Pickle file was not created")
            return

        # Second call should load directly from the pickle without re-processing
        print("\n2. Second call - loading from existing pickle...")
        chunks_second = save_or_load_pdf_chunks(pdf_chunks_path, temp_dir, simple_split_text)
        print(f"  ✓ Loaded {len(chunks_second)} chunk(s) from pickle")

        if chunks_first == chunks_second:
            print("  ✓ Chunks match between first and second calls")
        else:
            print("  ✗ Chunks mismatch between first and second calls")
            return

    except Exception as e:
        print(f"✗ Error testing save_or_load_pdf_chunks: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {temp_dir}")

if __name__ == "__main__":
    test_save_or_load_pdf_chunks()
    test_pdf_download_and_processing()
