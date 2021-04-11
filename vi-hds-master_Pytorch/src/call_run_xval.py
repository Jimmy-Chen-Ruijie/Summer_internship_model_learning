# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

from __future__ import absolute_import

import os
from src import procdata
import numpy as np
from src.run_xval import run_on_split, create_parser
from src.xval import XvalMerge
import src.utils as utils


def main():
    parser = create_parser(False)
    args = parser.parse_args()
    spec = utils.load_config_file(args.yaml)  # spec is a dict of dicts of dicts
    data_settings = procdata.apply_defaults(spec["data"])
    para_settings = utils.apply_defaults(spec["params"])

    xval_merge = XvalMerge(args, data_settings)
    print(xval_merge)
    for split_idx in range(1, args.folds + 1):
        print("---------------------------------------------------------------------------")
        print("    FOLD %d of %d" % (split_idx, args.folds))
        print("---------------------------------------------------------------------------")
        data_pair, val_results = run_on_split(args, data_settings, para_settings, split_idx, xval_merge.trainer)
        xval_merge.add(split_idx, data_pair, val_results)
    xval_merge.finalize()
    xval_merge.save()
    print('Completed')


if __name__ == "__main__":
    main()