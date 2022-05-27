#!/usr/bin/env python3

"""
Example usage (from a directory containing images):

  tag -l | python export_tags.py
"""

import csv
import os
import sys


def filename_is_valid(fn: str) -> bool:
    return os.path.isfile(fn) and not fn.startswith('.') and not fn.endswith('.csv')


def export_tags():
    all_tags_set = set()
    filename_tags = {}
    for line in sys.stdin:
        line = line.strip()
        if line:
            tokens = line.split('\t')
            if len(tokens) == 1:
                [filename] = tokens
                tags = []
            elif len(tokens) == 2:
                [filename, tags_str] = tokens
                tags = tags_str.split(',')
            else:
                raise Exception(f'line has unexpected number of pieces ({len(tokens)})')

            filename = filename.strip()
            tags = [t.strip() for t in tags]
            if filename_is_valid(filename):
                all_tags_set.update(tags)
                filename_tags[filename] = tags

    all_tags = sorted(all_tags_set)
    fieldnames = ['filename'] + all_tags
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for (filename, tags) in filename_tags.items():
        writer.writerow(dict(
            [('filename', filename)] +
            [(tag, '1' if tag in tags else '0') for tag in all_tags]
        ))


if __name__ == '__main__':
    export_tags()
