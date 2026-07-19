# End-to-end document ingestion pipeline
import os


def save_chunks(chunks, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        file_path = os.path.join(processed_dir, f"chunk_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)
