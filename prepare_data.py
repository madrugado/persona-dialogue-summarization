#!/usr/bin/env python3
"""
Clean HTML tags from dialogues.tsv and split into train/val/test (80:10:10).

The script:
1. Loads dialogues.tsv with HTML tags
2. Removes all HTML tags from persona_1_profile, persona_2_profile, and dialogue fields
3. Splits data into train (80%), val (10%), test (10%)
4. Saves splits to separate JSON files
"""

import argparse
import csv
import json
import os
import re
import random
from typing import List, Dict
from pathlib import Path


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Text with HTML tags

    Returns:
        Cleaned text without HTML tags
    """
    if not text or text.strip() == '':
        return ''

    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', text)
    # Replace HTML entities
    clean_text = re.sub(r'&nbsp;', ' ', clean_text)
    clean_text = re.sub(r'&lt;', '<', clean_text)
    clean_text = re.sub(r'&gt;', '>', clean_text)
    clean_text = re.sub(r'&amp;', '&', clean_text)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def parse_dialogue(dialogue_html: str) -> List[Dict]:
    """Parse dialogue from HTML format into list of messages.

    Args:
        dialogue_html: Dialogue text with HTML spans and <br /> tags

    Returns:
        List of message dicts with 'speaker' (1 or 2) and 'text'
    """
    if not dialogue_html or dialogue_html.strip() == '':
        return []

    messages = []

    # First, fix split <br /> tags within <span> elements
    def fix_split_brackets(text):
        # Replace <br /> inside <span> with a space
        pattern = r'<span class=participant_(\d+)>(.*?)</span>'
        result = []
        for match in re.finditer(pattern, text, re.DOTALL):
            speaker = int(match.group(1))
            content = match.group(2)
            # Replace <br /> inside content with space
            content = content.replace('<br />', ' ')
            # Clean up multiple spaces and newlines
            content = re.sub(r'\s+', ' ', content).strip()
            result.append((speaker, content))
        return result

    # Parse spans with their content
    span_messages = fix_split_brackets(dialogue_html)

    for speaker, raw_text in span_messages:
        # Split by "Пользователь X:" pattern within the same span
        # Pattern: "Пользователь X: message Пользователь X: message"
        user_pattern = r'Пользователь\s+(\d+):\s*'

        # Find all matches and their positions
        matches = list(re.finditer(user_pattern, raw_text))

        if not matches:
            # No "Пользователь X:" pattern, use the span's speaker
            text = raw_text
            text = clean_message_text(text)
            if text:
                messages.append({'speaker': speaker, 'text': text})
        else:
            # Process each message separated by "Пользователь X:"
            for i, match in enumerate(matches):
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
                text = raw_text[start:end].strip()

                # Use speaker from the "Пользователь X:" pattern
                msg_speaker = int(match.group(1))

                text = clean_message_text(text)
                if text:
                    messages.append({'speaker': msg_speaker, 'text': text})

    return messages


def clean_message_text(text: str) -> str:
    """Clean message text by removing artifacts.

    Args:
        text: Raw message text

    Returns:
        Cleaned message text
    """
    # Remove trailing artifacts like ")", "..", ",", "..."
    text = re.sub(r'\.\.\.\s*$', '', text)
    text = re.sub(r'\.\.\s*$', '', text)
    text = re.sub(r'\)\s*$', '', text)

    # Remove trailing comma with possible spaces before it (including " , " with spaces)
    text = re.sub(r'\s*,\s*$', '', text)

    # Remove duplicate colons at end
    text = re.sub(r':\s*$', '', text)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_dialogues(filepath: str) -> List[Dict]:
    """Load dialogues from TSV file.

    Args:
        filepath: Path to dialogues.tsv

    Returns:
        List of dialogue dictionaries
    """
    dialogues = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for i, row in enumerate(reader):
            persona_1 = remove_html_tags(row.get('persona_1_profile', ''))
            persona_2 = remove_html_tags(row.get('persona_2_profile', ''))
            dialogue_html = row.get('dialogue', '')

            messages = parse_dialogue(dialogue_html)

            if not messages:
                print(f"Warning: No messages found in dialogue {i}")
                continue

            dialogues.append({
                'id': i,
                'persona_1': persona_1,
                'persona_2': persona_2,
                'dialogue': messages,
            })

    return dialogues


def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """Split data into train, val, and test sets.

    Args:
        data: List of data points
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) lists
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def save_json(data: List[Dict], filepath: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} items to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean HTML from dialogues.tsv and split into train/val/test"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/dialogues.tsv",
        help="Input TSV file with dialogues",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Load dialogues
    print(f"Loading dialogues from {args.input}...")
    dialogues = load_dialogues(args.input)
    print(f"Loaded {len(dialogues)} dialogues")

    # Show a sample
    if dialogues:
        sample = dialogues[0]
        print("\nSample dialogue:")
        print(f"  Persona 1: {sample['persona_1'][:100]}...")
        print(f"  Persona 2: {sample['persona_2'][:100]}...")
        print(f"  Messages: {len(sample['dialogue'])}")
        for msg in sample['dialogue'][:3]:
            print(f"    Speaker {msg['speaker']}: {msg['text'][:50]}...")

    # Split data
    print(f"\nSplitting data {args.train_ratio}:{args.val_ratio}:{1 - args.train_ratio - args.val_ratio}...")
    train, val, test = split_data(dialogues, args.train_ratio, args.val_ratio, args.seed)

    print(f"Train: {len(train)} ({len(train)/len(dialogues)*100:.1f}%)")
    print(f"Val: {len(val)} ({len(val)/len(dialogues)*100:.1f}%)")
    print(f"Test: {len(test)} ({len(test)/len(dialogues)*100:.1f}%)")

    # Save splits
    output_dir = args.output_dir
    save_json(train, os.path.join(output_dir, "dialogues_train.json"))
    save_json(val, os.path.join(output_dir, "dialogues_val.json"))
    save_json(test, os.path.join(output_dir, "dialogues_test.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
