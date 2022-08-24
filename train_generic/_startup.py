import re
import os
import time
from collections import Counter
import yaml
import types
from types import GeneratorType
import scipy
import pickle
import shutil
import inspect
import numpy as np
import pandas as pd
from sys import platform
from pprint import pprint
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
matplotlib.use('agg')

getwd = os.getcwd

def setwd(path):
    owd = os.getcwd()
    os.chdir(path)
    return owd


##############################################################################
# GPU Management                                                             #
##############################################################################

MB = 1024 * 1024

gpus = nvidia = nvidia_smi = lambda: os.system('nvidia-smi')

def set_cuda_devices(i=""):
    """Set one or more GPUs to use for training by index or all by default
        Args:
            `i` may be a list of indices or a scalar integer index
                default='' # <- Uses all GPUs if you pass nothing
    """
    def list2csv(l):
        s = ''
        ll = len(l) - 1
        for i, x in enumerate(l):
            s += str(x)
            s += ',' if i < ll else ''
        return s 
    if i.__eq__(''): # Defaults to ALL
        i = list(range(DEV_COUNT))
    if isinstance(i, list):
        i = list2csv(i)

    # ensure other gpus not initialized by tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print("CUDA_VISIBLE_DEVICES set to {}".format(i))
    
def set_gpu_tf(gpu="", gpu_max_memory=None):
    """Set gpu for tensorflow upon initialization.  Call this BEFORE importing tensorflow"""
    set_cuda_devices(gpu)
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('\nUsed gpus:', gpus)
    if gpus:
        try:
            for gpu in gpus:
                print("Setting memory_growth=True for gpu {}".format(gpu))
                tf.config.experimental.set_memory_growth(gpu, True)
                if gpu_max_memory is not None:
                    print("Setting GPU max memory to: {} mB".format(gpu_max_memory))
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu, 
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_max_memory)]
                        )
        except RuntimeError as e:
            print(e)

def get_gpu_available_memory():
    return list(
        map(
            lambda x: N.nvmlDeviceGetMemoryInfo(
                N.nvmlDeviceGetHandleByIndex(x)).free // MB, range(DEV_COUNT)
            )
        )

def get_based_gpu_idx():
    mem_free = get_gpu_available_memory()
    idx = np.argmax(mem_free)
    print("GPU:{} has {} available MB".format(idx, mem_free[idx]))
    return idx

def set_based_gpu():
    idx = get_based_gpu_idx()
    set_gpu_tf(str(idx))


try:
    import pynvml as N
    N.nvmlInit()
    DEV_COUNT = N.nvmlDeviceGetCount()
    NVML_ERR = False
except:
    print("Exception caught in pynvml.nvmlInit()")
    DEV_COUNT = 0
    NVML_ERR = True



##############################################################################
# Environment Management                                                     #
##############################################################################
from types import GeneratorType

def inside_docker():
    path = '/proc/self/cgroup'
    x = (
        os.path.exists('/.dockerenv') or \
        os.path.isfile(path) and \
        any('docker' in line for line in open(path))
    )
    return any(list(x)) if isinstance(x, GeneratorType) else x

caller_id = get_caller_name = lambda: inspect.stack()[2][3]

def add_to_namespace(x, **kwargs):
    if not hasattr(x, '__dict__'):
        raise ValueError(
            "Cannot update nonexistant `__dict__` for object of type {}".format(type(x)))
    x.__dict__.update(kwargs)
    return x

def add_to_namespace_dict(x, _dict):
    x.__dict__.update(_dict)
    return x

def exists_here(object_str):
    if str(object_str) != object_str:
        print("Warning: Object passed in was not a string, and may have unexpected behvaior")
    return object_str in list(globals())

def stopifnot(predicate, **kwargs):
    locals().update(kwargs) # <-- inject necessary variables into local scope to check?
    predicate_str = predicate
    if is_strlike(predicate):
        predicate = eval(predicate)
    if is_bool(predicate) and predicate not in [True, 1]:
        import sys
        sys.exit("\nPredicate:\n\n  {}\n\n is not True... exiting.".format(
            predicate_str))

def add_to_globals(x):
    if type(x) == dict:
        if not all(list(map(lambda k: is_strlike(k), list(x)))):
            raise KeyError("dict `x` must only contain keys of type `str`")
    elif type(x) == list:
        if type(x[0]) == tuple:
            if not all(list(map(lambda t: is_strlike(t[0]), x))):
                raise ValueError("1st element of each tuple must be of type 'str'")
            x = dict(x)
        else:
            raise ValueError("`x` must be either a `list` of `tuple` pairs, or `dict`")
    globals().update(x)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __iter__(self):
        return iter(self.__dict__.items())
    def add_to_namespace(self, **kwargs):
        self.__dict__.update(kwargs)

def env(**kwargs):
    return Namespace(**kwargs)
environment = env

def parse_gitignore():
    with open('.gitignore', 'r') as f:
        x = f.readlines()
    ignore = list(map(lambda s: s.split('\n')[:-1], x))
    ignore[-1] = [x[-1]]
    return ', '.join(unlist(ignore))


is_function = lambda x: x.__class__.__name__ == 'function'
if_func = is_function

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

def which_os():
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macOS"
    elif platform == "win32":
        return "windows"
    else:
        raise ValueError("Mystery os...")

def on_windows():
    return which_os() == "windows"

def on_linux():
    return which_os() == "linux"

def on_mac():
    return which_os() == "macOS"

def import_flags(path=None):
    if path is not None:
        try:
            with open(path, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        except:
            pass
    possible_dirs = ['./', './config', './data', './fr_train']
    for directory in possible_dirs:
        path = os.path.join(directory, 'flags.yaml')    
        FLAGS_FILE = os.path.abspath(path)
        if os.path.exists(FLAGS_FILE):
            with open(FLAGS_FILE, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
    raise ValueError("No flags file found.")

def import_history(path='history/model_history'):
    with open(path, 'rb') as f:
      history = pickle.load(f)
    return history

def read_pickle(path):
    f = open(path, 'rb')
    x = pickle.load(f)
    f.close()
    return x


##############################################################################
# Math Operators                                                             #
##############################################################################
import math

add = lambda arr: reduce(lambda x, y: x + y, arr)
sub = lambda arr: reduce(lambda x, y: x - y, arr)
mul = lambda arr: reduce(lambda x, y: x * y, arr)
div = lambda arr: reduce(lambda x, y: x / y, arr)
prod = mul

def divisors(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def largest_divisor(n):
    return list(divisors(n))[-1]

##############################################################################
# List and Array Tools                                                       #
##############################################################################

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def is_bool(x):
    if x not in [True, False, 0, 1]:
        return False
    return True
isTrueOrFalse = is_bool

def is_strlike(x):
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def regextract(x, regex):
    matches = vmatch(x, regex)
    return np.asarray(x)[matches]
extract = find = regextract

def vmatch(x, regex):
    r = re.compile(regex)
    return np.vectorize(lambda x: bool(r.match(x)))(x)
rmatch = vmatch

def lengths(x):
    def maybe_len(e):
        if type(e) == list:
            return len(e)
        else:
            return 1
    if type(x) is not list: return [1]
    if len(x) == 1: return [1]
    return(list(map(maybe_len, x)))

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def next2pow(x):
    return 2**int(np.ceil(np.log(float(x))/np.log(2.0)))


def unnest(x, return_numpy=False):
    if return_numpy:
        return np.asarray([np.asarray(e).ravel() for e in x]).ravel()
    out = []
    for e in x:
        out.extend(np.asarray(e).ravel())
    return out
    
def unwrap_np(x):
    *y, = np.asarray(x, dtype=object)
    return y

def unwrap_df(df):
    if len(df.values.shape) >= 2:
        return df.values.flatten()
    return df.values

def df_diff(df1, df2):
	ds1 = set([tuple(line) for line in df1.values])
	ds2 = set([tuple(line) for line in df2.values])
	diff = ds1.difference(ds2)
	return pd.DataFrame(list(diff))

def summarize(x):
    x = np.asarray(x)
    x = np.squeeze(x)
    try:
        df = pd.Series(x)        
    except:
        try:
            df = pd.DataFrame(x)
        except:
            raise TypeError("`x` cannot be coerced to a pandas type.")
    return df.describe(include='all')

def list_product(els):
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod

def get_counts(df, colname):
    return pd.DataFrame.from_dict(
        dict(
            list(
                map(
                    lambda x: (x[0], len(x[1])), 
                    df.groupby(colname)
                )
            )
        ),
        orient='index'
    )

def get_duplicates(array):
    c = Counter(array)
    return [k for k in c if c[k] > 1] 

def get_duplicates_gen(array):
    c = Counter()
    seen = set()
    for i in array: 
        c[i] += 1
        if c[i] > 1 and i not in seen:
            seen.add(i)
            yield i

def np_arr_to_py(x):
    x = np.unstack()
    return list(x)

def logical2idx(x):
    x = np.asarray(x)
    return np.arange(len(x))[x]
l2i = logical2idx

def get(x, f):
    if is_scalar(x):
        return x.iloc[logical2idx(f(x))] if is_pandas(x) else x[logical2idx(f(x))]
    x = np.asarray(x)
    return x.iloc[logical2idx(lmap(x, f))] if is_pandas(x) else x[logical2idx(lmap(x, f))]

def apply_pred(x, p):
    return list(map(lambda e: p(e), x))

def extract_mask(x, m):
    if len(x) != len(m):
        raise ValueError("Shapes of `x` and `m` must be equivalent.")
    return np.asarray(x)[logical2idx(m)]

def extract_cond(x, p):
    mask = list(map(lambda e: p(e), x))
    return extract_mask(x, mask)

def import_file(filepath, ext='.py'):
    import importlib.util
    if not os.path.exists(filepath):
        raise ValueError("source `filepath` not found.")
    path = os.path.abspath(filepath)
    spec = importlib.util.spec_from_file_location(
        os.path.basename(
            path[:-len(ext)]
        ), 
        path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def maybe_list_up(x):
    if len(np.asarray(x).shape) == 0:
        return [x]
    return x

def idx_of(arr_to_seek, vals_to_idx):
    if isinstance(vals_to_idx[0], str):
        return idx_of_str(arr_to_seek, vals_to_idx)
    vals_to_idx = maybe_list_up(vals_to_idx)
    nested_idx = list(
        map(
            lambda x: np.where(arr_to_seek == x),
            vals_to_idx
        )
    )
    return list(set(unnest(nested_idx)))

def idx_of_str(arr_to_seek, vals_to_idx):
    vals_to_idx = maybe_list_up(vals_to_idx)
    arr_to_seek = np.asarray(arr_to_seek)
    nested_idx = list(
        map(
            lambda x: np.where(x == arr_to_seek), 
            vals_to_idx
        )
    )
    return list(set(unnest(nested_idx)))


def where(arr, elem, op='in'):
    return list(
        map(
            lambda x: elem == x if 'eq' in op else elem in x,
            arr
        )
    )

index = lambda arr, elem: unlist(logical2idx(where(arr, elem)))

def search(x, e):
    """ Look for and return (if found) element `e` in data structure `x`"""
    match = vmatch if type(list(x)[0]) == str else where
    ret = unnest(logical2idx(match(list(x), e)))
    if ret == []:
        return False
    return x[e] if type(x) == dict else x[ret[0]]

apply_df = lambda df, col, f: dict(zip(df[col].unique(), list(
    map(lambda i: f(df[col] == i), df[col].unique()))))

# Return a dict mapping of results
count_df = lambda df, col: apply_df(df, col, count)

def count(x):
    """Returns number of `True` elements of `x`"""
    return sum(np.asarray(x).astype(bool))

def counts(arr, x):
    """Returns the count of element `x` in `arr`"""
    arr = np.asarray(arr)
    return len(np.where(arr == x)[0])

def how_many(e, x):
    """Count how many of element `e` are in array `x`"""
    return count(np.asarray(x) == e)

def zipd(one, two):
    """zip dictionary"""
    return {**one, **two}

def maybe_unwrap(x):
    if hasattr(x, '__len__'):
        head, *_ = x
        return head
    else: 
        return x

def dict_list(keys):
    return {k: [] for k,v in dict.fromkeys(keys).items()}
    
def flip_kv(d):
    return {v: k for k,v in d.items()}

def condense(x):
    return np.concatenate(list(x.values()) if isinstance(x, dict) else x)

def within(x, y, eps=1e-3):
    ymax = y + eps
    ymin = y - eps
    return x <= ymax and x >= ymin

def within1(x, y):
    return within(x, y, 1.)

def within_vec(x, y, eps=1e-3):
    vf = np.vectorize(within)
    return np.all(vf(x, y, eps=eps))

def dim(x):
    if is_numpy(x):
        return x.shape
    return np.asarray(x).shape

def shapes(x):
    shapes_fun = FUNCS[type(x)]
    return shapes_fun(x)

def shapes_list(l, print_=False):
    r"""Grab shapes from a list of tensors or numpy arrays"""
    shps = []
    for x in l:
        if print_:
            print(np.asarray(x).shape)
        shps.append(np.asarray(x).shape)
    return shps

def sort_dict(d, by='key', rev=False):
    if 'v' in by:
        return {k: d[k] for k in sorted(d, key=d.get, reverse=rev)}
    return {k: d[k] for k in sorted(d, reverse=rev)}

def shapes_dict(d, print_=False):
    r"""Recursively grab shapes from potentially nested dictionaries"""
    shps = {}
    for k,v in d.items():
        if isinstance(v, dict):
            shps.update(shapes(v))
        else:
            if print_:
                print(k, ":\t", np.asarray(v).shape)
            shps[k] = np.asarray(v).shape
    return shps

def shapes_tuple(tup, return_shapes=False):
    shps = {i: None for i in range(len(tup))}
    for i, t in enumerate(tup):
        shps[i] = np.asarray(t).shape
    print(shps)
    if return_shapes:
        return shps

FUNCS = {
    dict: shapes_dict,
    list: shapes_list,
    tuple: shapes_tuple
}

def list_to_number_string(value):
    if isinstance(value, (list, tuple)):
        return str(value)[1:-1]
    else:
        return value

if False:
    df_list2numstr = lambda df, badcol: df[badcol].apply(list_to_number_string)
    mydf[badcol] = mydf[badcol].apply(list_to_number_string)

    import numpy as np
    import pandas as pd
    import scipy.sparse as sparse

    df = pd.DataFrame(np.arange(1,10).reshape(3,3))
    sparse.coo_matrix((nt), shape=(41,))
    arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
    df['newcol'] = arr.toarray().tolist()
    print(df)


def info(d, return_dict=False, print_=True):
    r"""Recursively grab shape, dtype, and size from (nested) dictionary of tensors"""
    info_ = {}
    for k,v in d.items():
        if isinstance(v, dict):
            info_.update(info(v))
        else:
            info_[k] = {
                'size': np.asarray(v).ravel().shape,
                'shape' :np.asarray(v).shape,
                'dtype': np.asarray(v).dtype.name
            }
            if print_:
                _v = np.asarray(v)
                print('key   -', k)
                print('dtype -', _v.dtype.name)
                print('size  -', np.asarray(v).ravel().shape)
                print('shape -', _v.shape)
                print()
    if return_dict:
        return info_


def stats(x, axis=None, epsilon=1e-7):
    if not is_numpy(x):
        x = np.asarray(x)
    if np.min(x) < 0:
        _x = x + abs(np.min(x) - epsilon)
    else:
        _x = x
    gmn = scipy.stats.gmean(_x, axis=axis)
    hmn = scipy.stats.hmean(_x, axis=axis)
    mode = scipy.stats.mode(x, axis=axis).mode[0]
    mnt2, mnt3, mnt4 = scipy.stats.moment(x, [2,3,4], axis=axis)
    lq, med, uq = scipy.stats.mstats.hdquantiles(x, axis=axis)
    lq, med, uq = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    var = scipy.stats.variation(x, axis=axis) # coefficient of variation
    sem = scipy.stats.sem(x, axis=axis) # std error of the means
    res = scipy.stats.describe(x, axis=axis)
    nms = ['nobs          ', 
           'minmax        ', 
           'mean          ', 
           'variance      ', 
           'skewness      ', 
           'kurtosis      ']
    description = dict(zip(nms, list(res)))
    description.update({
        'coeff_of_var  ': var,
        'std_err_means ': sem,
        'lower_quartile': lq,
        'median        ': med,
        'upper_quartile': uq,
        '2nd_moment    ': mnt2,
        '3rd_moment    ': mnt3,
        '4th_moment    ': mnt4,
        'mode          ': mode,
        'geometric_mean': gmn,
        'harmoinc_mean ': hmn
    })
    return description


def unzip(x):
    if type(x) is not list:
        raise ValueError("`x` must be a list of tuple pairs")
    return list(zip(*x))

from itertools import zip_longest
def groupl(iterable, n, padvalue=None, return_list=False):
  "groupl(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  x = zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
  return list(x) if return_list else x

def merge_by_colname(df1, df2, colname='target', how='outer'):
    pt = df1.pop(colname)
    nt = df2.pop(colname)    
    targets = pt.append(nt).reset_index()
    df2 = df1.merge(df2, how=how)
    df2.loc[:, (colname)] = targets
    return df2

from functools import reduce
reduce_df = lambda dfs: reduce(lambda x, y: merge_by_colname(x, y), dfs)
merge_dict = lambda dicts: reduce(
    lambda x,y: {k: v + [y[k]] for k,v in x.items()}, 
    dicts, 
    {k: [] for k in dicts[0].keys()}
)


def copy_dirtree(inpath, outpath):
    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    shutil.copytree(inpath, outpath, ignore=ignore_files)
    print("Success copying directory structure\n {} \n -- to --\n {}".format(
        inpath, outpath)
    )

def factors(n):
    """ Returns all factors of `n`. FAST"""
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def switch(on, pairs, default=None):
    """ Create dict switch-case from key-word pairs, mimicks R's `switch()`

        Params:
            on: key to index OR predicate returning boolean value to index into dict
            pairs: dict k,v pairs containing predicate enumeration results
        
        Returns: 
            indexed item by `on` in `pairs` dict
        Usage:
        # Predicate
            pairs = {
                True: lambda x: x**2,
                False: lambda x: x // 2
            }
            switch(
                1 == 2, # predicate
                pairs,  # dict 
                default=lambda x: x # identity
            )

        # Index on value
            key = 2
            switch(
                key, 
                values={1:"YAML", 2:"JSON", 3:"CSV"},
                default=0
            )
    """
    if type(pairs) is tuple:
        keys, vals = unzip(pairs)
        return switch2(on, keys=keys, vals=vals, default=default)
    if type(pairs) is not dict:
        raise ValueError("`pairs` must be a list of tuple pairs or a dict")
    return pairs.get(on, default)


def switch2(on, keys, vals, default=None):
    """
    Usage:
        switch(
            'a',
            keys=['a', 'b', 'c'],
            vals=[1, 2, 3],
            default=0
        )
        >>> 1

        # Can be used to select functions
        x = 10
        func = switch(
            x == 10, # predicate
            keys=[True, False],
            vals=[lambda x: x + 1, lambda x: x -1],
            default=lambda x: x # identity
        )
        func(x)
        >>> 11
    """
    if len(keys) == len(vals):
        raise ValueError("`keys` must be same length as `vals`")
    tuples = dict(zip(keys, vals))
    return tuples.get(on, default)


def comma_sep_str_to_int_list(s):
  return [int(i) for i in s.split(",") if i]

# Useful for printing all arguments to function call, mimicks dots `...` in R.
def printa(*argv):
    [print(i) for i in argv]

def printk(**kwargs):
    [print(k, ":\t", v) for k,v in kwargs.items()]

def pyrange(x, return_type='dict'):
    return {'min': np.min(x), 'max': np.max(x)} \
        if return_type == 'dict' \
        else (np.min(x), np.max(x))

def alleq(x):
    try:
        iter(x)
    except:
        x = [x]
    current = x[0]
    for v in x:
        if v != current:
            return False
    return True

def types(x):
    if isinstance(x, dict):
        return {k: type(v) for k,v in x.items()}
    return list(map(type, x))

# nearest element to `elem` in array `x`
def nearest1d(x, elem):
    lnx = len(x)
    if lnx % 2 == 1:
        mid = (lnx + 1) // 2
    else:
        mid = lnx // 2
    if mid == 1:
        return x[0]
    if x[mid] >= elem:
        return nearest1d(x[:mid], elem)
    elif x[mid] < elem:
        return nearest1d(x[mid:], elem)
    else:
        return x[0]

def idx_of_1d(x, elem):
    if elem not in x:
        raise ValueError("`elem` not contained in `x`")
    return dict(zip(x, range(len(x))))[elem]

def unlist(x):
    return list(map(lambda l: maybe_unwrap(l), x))

import random 
def random_ints(length, lo=-1e4, hi=1e4):
    return [random.randint(lo, hi) for _ in range(length)]
randint = random_ints

def random_floats(length, lo=-1e4, hi=1e4):
    return [random.random() for _ in range(length)]
randfloat = random_floats

def is_pandas(x):
    return x.__class__ in [
        pd.core.frame.DataFrame,
        pd.core.series.Series
    ]

def replace_at(x, indices, repl, colname=None):
    if is_pandas(x) and colname is None:
        if x.__class__ == pd.core.frame.DataFrame:
            raise ValueError("Must supply colname with a DataFrame object")
        return replace_at_pd(x, indices, repl)
    x = np.asarray(x)
    x[indices] = np.repeat(repl, len(indices))
    return x

def replace_at_pd(x, colname, indices, repl):
    x.loc[indices, colname] = repl
    return x

def delete_at(x, at):
    return x[~np.isin(np.arange(len(x)), at)]

def delete_at2(x, at):
    return x[[z for z in range(len(x)) if not z in at]]

def find_and_replace(xs, e, r):
    return replace_at(xs, np.where(xs == e)[0], r)

def _complex(real, imag):
    """ Efficiently create a complex array from 2 floating point """
    real = np.asarray(real)
    imag = np.asarray(imag)
    cplx = 1j * imag    
    return cplx + real

# Safe log10
def log10(data):
    np.seterr(divide='ignore')
    data = np.log10(data)
    np.seterr(divide='warn')
    return data
log10_safe = log10

def is_scalar(x):
    if is_numpy(x):
        return x.ndim == 0
    if isinstance(x, str) or type(x) == bytes:
        return True
    if hasattr(x, "__len__"):
        return len(x) == 1
    try:
        x = iter(x)
    except:
        return True
    return np.asarray(x).ndim == 0

def first(x):
    if is_scalar(x):
        return x
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[0]

def last(x):
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[-1] 

def listmap(x, f):
    return list(map(f, x))
lmap = listmap


##############################################################################
# Plotting Routines                                                          #
##############################################################################

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

# ALL_COLORS = list(colors.CSS4_COLORS)
COLORS = [
    'blue', # for original signal
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
] * 2


def histogram(x, bins='auto', show=True, save=False, outpath='histogram.png'):
    x = np.asarray(x).ravel()
    hist, bins = np.histogram(x, bins=bins)
    plt.bar(bins[:-1], hist, width=1)
    plt.savefig(outpath)
    if show:
        plt.show()

def plotx(y, x=None, xlab='obs', ylab='value', 
          title='', save=False, filepath='plot.png'):
    if x is None: x = np.linspace(0, len(y), len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab,
        title=title)
    ax.grid()
    if save:
       fig.savefig(filepath)
    plt.show()

def poverlap(y, x=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))
    for i in range(shp[0]):
        plt.plot(x, y[i], COLORS[i])
    plt.show()

# 2 cols per row
def psubplot(y, x=None, figsize=[6.4, 4.8], filename=None, title=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    _, ax = plt.subplots(nrows=shp[0]//2+1, ncols=2, figsize=figsize)
    for row in ax:
        for col in row:
            if i >= shp[0]: break
            col.plot(x, y[i], COLORS[i])
            i += 1
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# one col per row
def psub1(y, x=None, figsize=[6.4, 4.8], filename=None, title=None, xlab=None, ylab=None, hspace=0.5):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    fig, ax = plt.subplots(nrows=shp[0], ncols=1, figsize=figsize)
    for row in ax:
        if i >= shp[0]: break
        row.plot(x, y[i], COLORS[i])
        i += 1
    fig.subplots_adjust(hspace=hspace)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        ax[0].set_title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

psub = psubplot
polp = poverlap

def specgram(x, fs=1.0):
    from scipy import signal
    if not is_numpy(x):
        x = np.asarray(x)
    onesided = x.dtype.name not in ['complex64', 'complex128']
    f, t, Sxx = signal.spectrogram(x, fs, return_onesided=onesided)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.show()


##############################################################################
# Algorithms                                                                 #
##############################################################################
