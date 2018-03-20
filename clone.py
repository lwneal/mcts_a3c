#!/usr/bin/env python
# Creates a snapshot copy of this repository in target_dir
# The copy should include any uncommitted or unstaged changes
# (This makes experiments reproducible, even if they're not checked in)
import argparse
import os
import sys
import uuid
import os
import subprocess
import shutil

def mkdirp(path):
    os.makedirs(path, exist_ok=True)


def copy_repo(target_dir):
    # Get the list of tracked filenames in the current repo
    stdout = subprocess.check_output(['git', 'ls-files'])
    filenames = str(stdout, 'utf-8').splitlines()
    for src_filename in filenames:
        dst_filename = os.path.join(target_dir, src_filename)
        mkdirp(os.path.dirname(dst_filename))
        shutil.copy2(src_filename, dst_filename)
    print('Copied {} files to {}'.format(len(filenames), target_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', required=True, help='A helpful description so that a month from now, you can remember why you ran this experiment.')
    options = vars(parser.parse_args())

    dataset_name = 'baby-a3c'
    random_hex = uuid.uuid4().hex[:8]
    result_dir = '/mnt/nfs/experiments/{}_{}'.format(dataset_name, random_hex)
    copy_repo(result_dir)
