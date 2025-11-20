import re
import csv

INPUT_FILE = "data/ppo/ppo_out.txt"
OUTPUT_FILE = "data/ppo/ppo_metrics.csv"


def parse_block(block_text):
    """
    Parse a single block like:

    | key | value |

    Handles empty value fields.
    """
    data = {}
    # Matches lines like: |    entropy   | -1.46 |
    line_re = re.compile(r"\|\s*(.*?)\s*\|\s*(.*?)\s*\|")

    for line in block_text.strip().split("\n"):
        m = line_re.match(line)
        if not m:
            continue

        key, value = m.groups()

        # Remove trailing slash categories (like bc/ , rollout/)
        if key.endswith("/"):
            key = key[:-1]

        key = key.strip()

        # Convert to float/int when possible
        value = value.strip()
        if value == "":
            data[key] = None
        else:
            try:
                if "." in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                data[key] = value

    return data


def extract_all_blocks(text):
    """Return a list of dicts for each block."""
    # Blocks are separated by lines of dashes
    raw_blocks = re.split(r"-{20,}", text)
    blocks = []

    for blk in raw_blocks:
        if "|" not in blk:
            continue
        parsed = parse_block(blk)
        if parsed:
            blocks.append(parsed)

    return blocks


if __name__ == "__main__":
    # Load file
    with open(INPUT_FILE, "r") as f:
        text = f.read()

    blocks = extract_all_blocks(text)

    if not blocks:
        print("No metric blocks found.")
        exit()

    # Gather all possible keys (columns)
    all_keys = sorted({k for block in blocks for k in block.keys()})

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for block in blocks:
            writer.writerow(block)

    print(f"Extracted {len(blocks)} blocks into {OUTPUT_FILE}")
