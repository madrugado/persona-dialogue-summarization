#!/usr/bin/env python3
"""
Clean HTML tags from dialogues.tsv and split into train/val/test (80:10:10).
"""

import argparse
import csv
import json
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple


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
    return clean_text.strip()


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
            # No "Пользователь X:" pattern, use span's speaker
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

                # Use speaker from "Пользователь X:" pattern
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
    text = re.sub(r'\)\s*$', '', text)
    # Remove trailing comma with possible spaces before it (including " , " with spaces)
    text = re.sub(r'\s*,\s*$', '', text)
    # Remove duplicate colons at end
    text = re.sub(r':\s*$', '', text)

    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_persona_index(data: List[Dict]) -> Dict[str, Set[int]]:
    """Build an index mapping personas to dialogue IDs containing them.

    Args:
        data: List of dialogue dicts with 'id', 'persona_1', 'persona_2'

    Returns:
        Dictionary mapping persona text to set of dialogue IDs
    """
    persona_index: Dict[str, Set[int]] = {}
    for item in data:
        dialog_id = item['id']
        persona = item.get('persona_2', '')
        if persona:
                if persona not in persona_index:
                    persona_index[persona] = set()
                persona_index[persona].add(dialog_id)
    return persona_index


def _build_connected_component(
    persona_index: Dict[str, Set[int]],
    data_by_id: Dict[int, Dict],
    start_id: int,
) -> Tuple[Set[int], Set[str]]:
    """Find all dialogues and personas in the connected component starting from a dialogue.

    Connected components are built using persona_2 only to avoid all dialogues being linked.

    Args:
        persona_index: Mapping from persona_2 to dialogue IDs
        data_by_id: Mapping from dialogue ID to dialogue data
        start_id: Starting dialogue ID

    Returns:
        Tuple of (dialogue IDs in component, personas in component)
    """
    dialogue_ids = set([start_id])
    personas = set()
    queue = [start_id]

    while queue:
        dialog_id = queue.pop(0)
        dialog_data = data_by_id[dialog_id]

        # Add persona_2 from this dialogue
        persona = dialog_data.get('persona_2', '')
        if persona and persona not in personas:
            personas.add(persona)
            # Find all dialogues with this persona_2
            for related_id in persona_index.get(persona, set()):
                if related_id not in dialogue_ids:
                    dialogue_ids.add(related_id)
                    queue.append(related_id)

    return dialogue_ids, personas


def split_data(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, val, and test sets with no duplicate personas.

    Uses connected components to ensure each persona appears in at most one split.

    Args:
        data: List of dialogue dicts
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) lists
    """
    random.seed(seed)

    # Build persona index and data lookup
    persona_index = build_persona_index(data)
    data_by_id = {d['id']: d for d in data}

    # Calculate target sizes
    total = len(data)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    # Build splits using connected components
    remaining_ids: Set[int] = set(d['id'] for d in data)
    val_ids: List[int] = []
    test_ids: List[int] = []

    def add_to_split(target_ids: List[int], target_size: int) -> bool:
        """Try to add a connected component to a split.

        Returns True if component fits, False otherwise.
        """
        # Try to find a small component that fits
        shuffled_remaining = list(remaining_ids)
        random.shuffle(shuffled_remaining)

        sample_size = min(100, len(shuffled_remaining))
        sample_ids = shuffled_remaining[:sample_size]

        for dialog_id in sample_ids:
            if dialog_id not in remaining_ids:
                continue
            component_dialogues, _ = _build_connected_component(
                persona_index, data_by_id, dialog_id
            )
            if len(target_ids) + len(component_dialogues) <= target_size:
                target_ids.extend(component_dialogues)
                for d_id in component_dialogues:
                    remaining_ids.discard(d_id)
                return True
        return False

    # Build val split first
    while len(val_ids) < val_target and remaining_ids:
        if not add_to_split(val_ids, val_target):
            # Add one random dialogue if no component fits
            random_id = random.choice(list(remaining_ids))
            val_ids.append(random_id)
            remaining_ids.remove(random_id)

    # Build test split second
    test_target = total - train_target - val_target
    while len(test_ids) < test_target and remaining_ids:
        if not add_to_split(test_ids, test_target):
            # Add one random dialogue if no component fits
            random_id = random.choice(list(remaining_ids))
            test_ids.append(random_id)
            remaining_ids.remove(random_id)

    # Remaining go to train
    train_ids = list(remaining_ids)

    train = [data_by_id[pid] for pid in train_ids]
    val = [data_by_id[pid] for pid in val_ids]
    test = [data_by_id[pid] for pid in test_ids]

    return train, val, test


def save_json(data: List[Dict], filepath: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path

    Returns:
        None
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

    # Load dialogues from TSV
    print(f"Loading dialogues from {args.input}...")
    dialogues = []
    with open(args.input, 'r', encoding='utf-8') as f:
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

    # Split data into train/val/test
    print(f"Splitting data {args.train_ratio}:{args.val_ratio}:{1 - args.train_ratio - args.val_ratio}...")
    train, val, test = split_data(dialogues, args.train_ratio, args.val_ratio, args.seed)

    # Save splits
    output_dir = args.output_dir
    save_json(train, os.path.join(output_dir, "dialogues_train.json"))
    save_json(val, os.path.join(output_dir, "dialogues_val.json"))
    save_json(test, os.path.join(output_dir, "dialogues_test.json"))

    print("\nDone!")
    print(f"Train: {len(train)} ({len(train)/len(dialogues)*100:.1f}%)")
    print(f"Val: {len(val)} ({len(val)/len(dialogues)*100:.1f}%)")
    print(f"Test: {len(test)} ({len(test)/len(dialogues)*100:.1f}%)")

main()

