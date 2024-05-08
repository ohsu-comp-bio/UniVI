from typing import Union
from pathlib import Path, PosixPath
import numpy as np

import pandas as pd

def path2df(filepath: Union[str, PosixPath]) -> pd.DataFrame:

    if not isinstance(filepath, PosixPath):
        filepath = Path(filepath)

    ''' handle file type '''
    if filepath.suffix == ".csv":
        return pd.read_csv(filepath, header=0, index_col=0)
    else:
        raise TypeError("unknown file type")

