import pandas as pd
from pathlib import Path
import pytesseract
from PIL import Image
import re
from tqdm import tqdm

def extract_text(image_path):
    #Extract text with OCR
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        # Minimal cleaning - just normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return ""

def create_summary(article):
    #extractive summary
    # Split into sentences
    sentences = re.split(r'[.!?]+', article)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if len(sentences) <= 3:
        return article

    # Keep 30% of sentences
    num_keep = max(3, int(len(sentences) * 0.30))

    # Select: first + evenly distributed + last
    selected = [sentences[0]]

    if num_keep > 2:
        step = len(sentences) // (num_keep - 1)
        for i in range(1, num_keep - 1):
            idx = i * step
            if idx < len(sentences) - 1:
                selected.append(sentences[idx])

    selected.append(sentences[-1])

    return ' '.join(selected)

def validate(article, summary):
    #Validation article-summary pair
    article_words = len(article.split())
    summary_words = len(summary.split())

    # Thresholds
    if article_words < 80 or article_words > 2000:
        return False

    if summary_words < 25 or summary_words > 200:
        return False

    # Compression ratio: 15-45%
    compression = summary_words / article_words
    if not (0.15 <= compression <= 0.45):
        return False

    return True

# Main processing
print("🔍 Scanning for images...\n")

image_folder = Path("newspaper_images")
images = (
    list(image_folder.glob("*.jpg")) +
    list(image_folder.glob("*.JPG")) +
    list(image_folder.glob("*.jpeg")) +
    list(image_folder.glob("*.JPEG")) +
    list(image_folder.glob("*.png")) +
    list(image_folder.glob("*.PNG"))
)

print(f"📸 Found {len(images)} images\n")
print("🔄 Processing (OCR + Summarization)...\n")

# Process all images
valid_pairs = []
failed_images = []

for img_path in tqdm(images, desc="Processing", unit="image"):
    # Extract text
    article = extract_text(img_path)

    if not article:
        failed_images.append({
            'image': img_path.name,
            'reason': 'OCR failed',
            'words': 0
        })
        continue

    # Create summary
    summary = create_summary(article)

    # Validate
    if not validate(article, summary):
        failed_images.append({
            'image': img_path.name,
            'reason': 'Validation failed',
            'words': len(article.split())
        })
        continue

    # Store valid pair
    valid_pairs.append({
        'Article': article,
        'Summary': summary,
        'source_image': img_path.name,
        'article_words': len(article.split()),
        'summary_words': len(summary.split()),
        'compression_ratio': round(len(summary.split()) / len(article.split()), 3)
    })

# Display results

print("📊 PROCESSING RESULTS")
print(f"{'='*80}")
print(f"Total images: {len(images)}")
print(f"Valid pairs: {len(valid_pairs)}")
print(f"Failed: {len(failed_images)}")
print(f"Success rate: {len(valid_pairs)/len(images)*100:.1f}%")

if len(valid_pairs) > 0:
    df = pd.DataFrame(valid_pairs)

    print(f"\n Quality Metrics:")
    print(f"  Average article length: {df['article_words'].mean():.0f} words")
    print(f"  Average summary length: {df['summary_words'].mean():.0f} words")
    print(f"  Average compression: {df['compression_ratio'].mean():.2f}")
    print(f"  Min article: {df['article_words'].min()} words")
    print(f"  Max article: {df['article_words'].max()} words")

    # Create output folder
    output_folder = Path("dataset_output")
    output_folder.mkdir(exist_ok=True)

    # Save datasets
    print(f"\n Saving datasets...\n")

    # Training-ready CSV (Article + Summary only)
    training_csv = output_folder / "editorial_dataset_training.csv"
    df[['Article', 'Summary']].to_csv(training_csv, index=False)
    print(f"  Training dataset: {training_csv}")
    print(f"     ({len(df)} pairs ready for fine-tuning)")

    # Full CSV with metadata
    full_csv = output_folder / "editorial_dataset_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"   Full dataset: {full_csv}")
    print(f"     (includes metadata: word counts, compression ratios)")

    # Save failed images log
    if failed_images:
        failed_csv = output_folder / "failed_images.csv"
        pd.DataFrame(failed_images).to_csv(failed_csv, index=False)
        print(f"  ⚠️  Failed images: {failed_csv}")

    # Generate quality report
    report_path = output_folder / "dataset_quality_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EDITORIAL SUMMARIZATION DATASET - QUALITY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: 2025-11-19 05:40:38 UTC\n")
        f.write(f"Author: ZabeeAhamed\n")
        f.write(f"Project: Editorial Summarization System\n")
        f.write("="*80 + "\n\n")

        f.write("DATASET STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total images processed: {len(images)}\n")
        f.write(f"Valid article-summary pairs: {len(valid_pairs)}\n")
        f.write(f"Failed extractions: {len(failed_images)}\n")
        f.write(f"Success rate: {len(valid_pairs)/len(images)*100:.1f}%\n\n")

        f.write(" QUALITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Article Length (words):\n")
        f.write(f"  Min: {df['article_words'].min()}\n")
        f.write(f"  Max: {df['article_words'].max()}\n")
        f.write(f"  Mean: {df['article_words'].mean():.1f}\n")
        f.write(f"  Median: {df['article_words'].median():.1f}\n\n")

        f.write(f"Summary Length (words):\n")
        f.write(f"  Min: {df['summary_words'].min()}\n")
        f.write(f"  Max: {df['summary_words'].max()}\n")
        f.write(f"  Mean: {df['summary_words'].mean():.1f}\n")
        f.write(f"  Median: {df['summary_words'].median():.1f}\n\n")

        f.write(f"Compression Ratio:\n")
        f.write(f"  Min: {df['compression_ratio'].min():.3f}\n")
        f.write(f"  Max: {df['compression_ratio'].max():.3f}\n")
        f.write(f"  Mean: {df['compression_ratio'].mean():.3f}\n")
        f.write(f"  Median: {df['compression_ratio'].median():.3f}\n\n")

        f.write(" SAMPLE OUTPUTS (First 3 Pairs)\n")
        f.write("="*80 + "\n\n")

        for idx, row in df.head(3).iterrows():
            f.write(f"Sample {idx + 1}:\n")
            f.write(f"Image: {row['source_image']}\n")
            f.write(f"Article ({row['article_words']} words):\n")
            f.write(f"{row['Article'][:400]}...\n\n")
            f.write(f"Summary ({row['summary_words']} words):\n")
            f.write(f"{row['Summary']}\n")
            f.write(f"Compression: {row['compression_ratio']}\n")
            f.write("-"*80 + "\n\n")

        f.write(" ASSESSMENT\n")
        f.write("="*80 + "\n")
        if len(valid_pairs) >= 150:
            f.write(" EXCELLENT: Dataset is ready for fine-tuning!\n")
            f.write(f"   {len(valid_pairs)} pairs provide strong training signal.\n")
        elif len(valid_pairs) >= 100:
            f.write(" GOOD: Dataset is sufficient for fine-tuning.\n")
        elif len(valid_pairs) >= 50:
            f.write("⚠️ MODERATE: Dataset is usable but limited.\n")
        else:
            f.write(" INSUFFICIENT: More data needed.\n")

        f.write("\n NEXT STEPS\n")
        f.write("-"*80 + "\n")
        f.write("1. Review sample outputs for quality\n")
        f.write("2. Use editorial_dataset_training.csv for T5 fine-tuning\n")
        f.write("3. Split into train/validation/test (80/10/10)\n")
        f.write("4. Establish baseline with pretrained T5\n")
        f.write("5. Fine-tune and evaluate with ROUGE metrics\n")

    print(f"   Quality report: {report_path}")

    # Display sample
    print(f"\n SAMPLE OUTPUT (First Pair):")
    print(f"{'='*80}")
    sample = df.iloc[0]
    print(f"Image: {sample['source_image']}")
    print(f"Article ({sample['article_words']} words):")
    print(f"{sample['Article'][:300]}...")
    print(f"\nSummary ({sample['summary_words']} words):")
    print(f"{sample['Summary']}")
    print(f"Compression: {sample['compression_ratio']}")

    print(f"\n{'='*80}")
    print("🎉 DATASET CREATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nSUCCESS! Your dataset is ready for training.")
    print(f"\n Files created:")
    print(f"   • editorial_dataset_training.csv ({len(df)} pairs)")
    print(f"   • editorial_dataset_full.csv (with metadata)")
    print(f"   • dataset_quality_report.txt (detailed analysis)")
    print(f"\n Next: Fine-tune T5 model on this dataset!")
    print(f"{'='*80}\n")

else:
    print(f"\n NO VALID PAIRS CREATED!")
    print(f"\nCheck failed_images.csv for details.")

print("\n Processing complete.\n")