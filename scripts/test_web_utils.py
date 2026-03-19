#!/usr/bin/env python3
"""
Test script to verify web scraping and pickle saving functionality.
"""

import os
import pickle
from web_utils import scrape_webpage, save_or_load_web_chunks

def simple_split_text(text, chunk_size=500):
    """Simple text splitting function for testing."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def test_web_scraping_and_storage():
    print("=" * 80)
    print("Testing Web Scraping and Storage")
    print("=" * 80)
    
    # Test URLs - using some TU Delft research data management pages
    test_urls = [
        "https://www.tudelft.nl/en/library/research-data-management",
        # "https://www.tudelft.nl/en/library/data-management#c1634274",
    ]
    
    # Step 1: Test scraping individual webpages
    print("\n1. Testing individual webpage scraping...")
    for i, url in enumerate(test_urls, 1):
        try:
            print(f"\n  {i}. Scraping: {url}")
            text = scrape_webpage(url)
            text_length = len(text)
            lines = text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            print(f"     ✓ Successfully scraped")
            print(f"     - Total characters: {text_length:,}")
            print(f"     - Total lines: {len(lines):,}")
            print(f"     - Non-empty lines: {len(non_empty_lines):,}")
            print(f"     - First 300 characters:")
            print("     " + "-" * 70)
            preview = text[:300].replace('\n', ' ')
            print(f"     {preview}")
            print("     " + "-" * 70)
        except Exception as e:
            print(f"     ✗ Error scraping webpage: {e}")
            return
    
    # Step 2: Test chunking
    print("\n2. Testing text chunking...")
    try:
        sample_text = scrape_webpage(test_urls[0])
        chunks = simple_split_text(sample_text, chunk_size=500)
        print(f"  ✓ Text split into {len(chunks)} chunks")
        print(f"  - Average chunk size: {sum(len(c) for c in chunks) // len(chunks):,} characters")
        print(f"  - First chunk preview (first 200 chars):")
        print("  " + "-" * 70)
        print(f"  {chunks[0][:200].replace(chr(10), ' ')}")
        print("  " + "-" * 70)
    except Exception as e:
        print(f"  ✗ Error during chunking: {e}")
        return
    
    # Step 3: Test pickle saving with save_or_load_web_chunks
    print("\n3. Testing save_or_load_web_chunks utility function...")
    pickle_path = "./test_web_chunks.pkl"
    
    try:
        # First call - should scrape and create pickle
        print("  First call (scraping and creating pickle)...")
        chunks1 = save_or_load_web_chunks(
            pickle_path, 
            test_urls, 
            simple_split_text
        )
        print(f"  ✓ Created pickle with {len(chunks1)} chunks")
        print(f"  - Total characters in all chunks: {sum(len(c) for c in chunks1):,}")
        
        # Verify pickle file was created
        if os.path.exists(pickle_path):
            pickle_size = os.path.getsize(pickle_path)
            print(f"  ✓ Pickle file created: {pickle_path} ({pickle_size:,} bytes)")
        else:
            print(f"  ✗ Pickle file was not created!")
            return
        
        # Second call - should load from pickle
        print("\n  Second call (loading from pickle)...")
        chunks2 = save_or_load_web_chunks(
            pickle_path, 
            test_urls, 
            simple_split_text
        )
        print(f"  ✓ Loaded {len(chunks2)} chunks from pickle")
        
        # Verify integrity
        if chunks1 == chunks2:
            print(f"  ✓ Pickle load successful - chunks match exactly")
        else:
            print(f"  ✗ Pickle load failed - chunks mismatch!")
            print(f"    - First call: {len(chunks1)} chunks")
            print(f"    - Second call: {len(chunks2)} chunks")
            return
    except Exception as e:
        print(f"  ✗ Error with save_or_load_web_chunks: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test manual pickle operations
    print("\n4. Testing manual pickle save/load...")
    manual_pickle_path = "./test_web_manual.pkl"
    
    try:
        sample_data = {
            "urls": test_urls,
            "chunks": chunks1[:5],  # Save first 5 chunks
            "metadata": {
                "total_chunks": len(chunks1),
                "total_chars": sum(len(c) for c in chunks1)
            }
        }
        
        # Save
        with open(manual_pickle_path, "wb") as f:
            pickle.dump(sample_data, f)
        manual_pickle_size = os.path.getsize(manual_pickle_path)
        print(f"  ✓ Manually saved pickle: {manual_pickle_path} ({manual_pickle_size:,} bytes)")
        
        # Load
        with open(manual_pickle_path, "rb") as f:
            loaded_data = pickle.load(f)
        
        # Verify
        if (loaded_data["urls"] == sample_data["urls"] and 
            loaded_data["chunks"] == sample_data["chunks"] and
            loaded_data["metadata"] == sample_data["metadata"]):
            print(f"  ✓ Manual pickle operations successful")
            print(f"    - URLs: {len(loaded_data['urls'])}")
            print(f"    - Chunks: {len(loaded_data['chunks'])}")
            print(f"    - Total chars: {loaded_data['metadata']['total_chars']:,}")
        else:
            print(f"  ✗ Manual pickle data mismatch!")
    except Exception as e:
        print(f"  ✗ Error with manual pickle operations: {e}")
        return
    
    print("\n" + "=" * 80)
    print("All tests completed successfully! ✓")
    print("=" * 80)
    
    # Cleanup info
    print("\nTest artifacts created:")
    print(f"  - {pickle_path}")
    print(f"  - {manual_pickle_path}")
    print("\nTo clean up test files, run:")
    print(f"  rm -f {pickle_path} {manual_pickle_path}")

if __name__ == "__main__":
    test_web_scraping_and_storage()
