# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import dataprep as dp
import numpy as np

@click.command()
@click.argument('input_projectdir', type=click.Path(exists=True))
@click.option('--traincnt', default = 2048)
@click.option('--testcnt', default = 1024)
def main(input_projectdir, traincnt, testcnt):
    if traincnt == 0 or testcnt == 0:
        return

    dfFilePath = os.environ.get("REF_RAW_TRAIN_DF")
    catColName = os.environ.get("CATEGORY_COLUMN_NAME")
    procDfTrainFilePath = os.path.join(input_projectdir, os.environ.get("REF_PROC_TRAIN_DF"))
    procDfTestFilePath = os.path.join(input_projectdir, os.environ.get("REF_PROC_TEST_DF"))

    destTestOrigDir = os.path.join(input_projectdir, os.environ.get("PROC_TEST_ORIG_DIR"))
    destTestAugmDir = os.path.join(input_projectdir, os.environ.get("PROC_TEST_AUG_DIR"))
    destTrainOrigDir = os.path.join(input_projectdir, os.environ.get("PROC_TRAIN_ORIG_DIR"))
    destTrainAugmDir = os.path.join(input_projectdir, os.environ.get("PROC_TRAIN_AUG_DIR"))

    df = pd.read_csv(dfFilePath)
    catv = df[catColName].unique()
    catv = np.array([3,4])
    dataFramePrep = dp.DataFramePreparation(traincnt, procDfTrainFilePath, destTrainOrigDir, destTrainAugmDir,
                                         testcnt, procDfTestFilePath, destTestOrigDir, destTestAugmDir)
    dataFramePrep(catv)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()




