import os
import scipy
import inspect
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import Tensor, EagerTensor
from tensorflow.python.framework.config import list_physical_devices
import seaborn as sns
import yaml
import pickle
from types import GeneratorType
import time 


timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


def binary_weighted_crossentropy(weight, use_tf=True):
    """ Compute weighted binary crossentropy using tensorflow routines from logits
    params:
        weight: int
            weight > 1 decreases false negative count and increases recall
            weight < 1 decreases false positive count and increases precision
    """
    def binary_weighted_crossentropy_internal(labels, logits):
        if 'int' in labels.dtype.name:
            labels = tf.cast(labels, 'float32')
        if use_tf:
            return tf.nn.weighted_cross_entropy_with_logits(labels, logits, weight)
        x, z, q = logits, labels, weight
        return (1 - z) * x + (1 + (q - 1) * z) * tf.math.log(1 + tf.exp(-x))
    return binary_weighted_crossentropy_internal

def weighted_binary_crossentropy(w1, w2):
    """
    Computes weighted binary crossentropy
    params:
        w1, w2: the weights for the two classes, (0, 1) respectively
    Usage:
     model.compile(
         loss=weighted_binary_crossentropy(0.7, 0.3), optimizer="adam", metrics=["accuracy"])
    """
    if tf.round(w1 + w2) != 1.0:
        raise ValueError("`recall_weight` and `spec_weight` must sum to 1.")
    @tf.autograph.experimental.do_not_convert
    def loss(y_true, y_pred):
        # avoid absolute 0
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ones   = tf.ones_like(y_true)
        mask   = tf.equal(y_true, ones)
        res, _ = tf.map_fn(lambda x: (tf.multiply(-tf.math.log(x[0]), w1) if x[1] is True
                                      else tf.multiply(-tf.math.log(1 - x[0]), w2), x[1]),
                           (y_pred, mask), dtype=(tf.float32, tf.bool))
        return res
    return loss


def restore_model(model_path, init_compile=True, custom_objects=None, compile_kwargs=None):
    """ Load and compile saved tensorflow model objects from disk. It is recommended
        you use method A if possible, as it does not required named layers, metrics, or
        optimizers in order to associate function handles and is inherently less brittle.

    params: 
        model_path: str path to directory that contains .pb model file
        init_compile: bool whether to attempt to compile the model 'as-is' on load
        custom_objects: dict of object_name, object_handle pairs required to load custom model
        compile_kwargs: dict of kwargs to call compile() on the resultant (custom) model object.
    returns:
        tensorflow model object

    path = './models/aleatoric'

    # Method A - using `compile_kwargs` dict to compile after initial load
    args = {
        'optimizer': 'adam',
        'loss': {'logits_variance': bayesian_categorical_crossentropy(100, 1),
        'sigmoid_output': binary_weighted_crossentropy(1.5)}, 
        'metrics': {},
        'loss_weights': {'logits_variance': 0.2, 'sigmoid_output': 1.0}} 
    model = restore_model(path, compile_kwargs=args)

    # Method B - using `custom_objects` dict to compile on initial load
    num_monte_carlo = 100
    loss_weight = 1.5
    custom_objs = {
        'bayesian_categorical_crossentropy_internal': bayesian_categorical_crossentropy(num_monte_carlo, 1),
        'binary_weighted_crossentropy_internal': binary_weighted_crossentropy(loss_weight)}
    model = restore_model(model_path, init_compile=True, custom_objects=custom_objs)

    print(model.summary())
    """
    if not os.path.exists(model_path):
        raise ValueError("No model found at {}".format(model_path))
    try:
        model = tf.keras.models.load_model(
            model_path, compile=init_compile, custom_objects=custom_objects)
        if compile_kwargs is not None and init_compile is True:
            raise ValueError("Must set `init_compile=False` if passing `compile_args`")
        elif not init_compile:
            model.compile(**compile_kwargs)
    except :
        raise ImportError(
            "Error loading model {}".format(model_path))
    print(model.summary)
    return model



def inside_docker():
    path = '/proc/self/cgroup'
    x = (
        os.path.exists('/.dockerenv') or \
        os.path.isfile(path) and \
        any('docker' in line for line in open(path))
    )
    return any(list(x)) if isinstance(x, GeneratorType) else x

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

def get_src(f, return_lines=True):
    if return_lines:
        return inspect.getsourcelines(f)
    return inspect.getsource(f)
src=get_src

# Convenience func
lmap = lambda f, x: list(map(f, x))

# Function to unlist if there's an outer list
maybe_unlist = lambda x: x[0] if hasattr(x, "__len__") and len(x) == 1 else x

# Convert bool array to indices
logical2idx = lambda x: np.arange(len(x))[np.asarray(x)]

# Putting it all together
where = lambda x, fx, p: maybe_unlist(
    logical2idx(lmap(f=lambda e: p(fx(e)), x=x))
)

# Get nice and easy from where we just wrote
get = lambda x, fx, p: np.asarray(x)[where(x, fx, p)]

dslen = lambda x: len(list(x.as_numpy_iterator()))

def _dir(path='./'):
    return [x[0] for x in os.walk(path)][1:]

def maybe_undict(d):
    return tuple(d.values()) if isinstance(d, dict) else d

def nonzero(x):
    return not (np.all(x) == 0.0)

nonzero = lambda x: any(list(
    map(lambda y: np.count_nonzero(y) > 0, x))) \
        if hasattr(x, "__len__") else any(np.count_nonzero(x))


def downsample(x, du=640, su=50):
    if 'tensorflow' in str(x.__class__):
        x = x.numpy()
    return scipy.signal.resample_poly(x, su, du)


def load_sig(path, return_fs=False):
    import soundfile as sf
    x, fs = sf.read(path)
    if x.ndim >= 2:
        x = x[:,0] + x[:,1]
    return x, fs

def stft_fix(x, win_len, axis=0):
    pre  = tf.zeros([win_len // 2], dtype=x.dtype)
    post = tf.zeros([tf.cast(tf.math.ceil(win_len / 2), 'int32')], x.dtype)
    return tf.concat([pre, x, post], axis=axis)

def grab(dataset):
    r"""Convenient but expensive way to quickly view a batch.
        Args:
            dataset: A tensorflow dataset object.
        Returns:
            nb: dict, a single batch of data, having forced evaluation of
            lazy map calls.
    """
    return next(dataset.as_numpy_iterator())

def is_strlike(x):
    if is_tensor(x):
        return x.dtype == tf.string
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def is_bool(x):
    if is_tensor(x):
        return x.dtype == tf.bool
    if x not in [True, False, 0, 1]:
        return False
    return True

def list_devices():
  return list_physical_devices()

def list_device_names(XLA=False):
    out = list(map(lambda x: x.name, list_devices()))
    if not XLA:
        out = [i for i in out if not ":XLA_" in i]
    return out

def count_gpus_available():
  x = list_device_names()
  return len(x) - 1

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def is_tensor(x):
    return x.__class__ in [Tensor, EagerTensor]

def is_complex_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_complex

def is_float_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_floating

def is_integer_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_integer

def as_tensor(x, dtype=None):
    if x is None: return x

    if type(dtype) == str:
        dtype = tf.as_dtype(dtype)

    if is_tensor(x) and not (dtype is None):
        return tf.cast(x, dtype)
    else:
        # this can do an overflow, but it'll issue a warning if it does
        # in that case, use tf$cast() instead of tf$convert_to_tensor, but
        # at that range precision is probably not guaranteed.
        # the right fix then is tf$convert_to_tensor('float64') %>% tf$cast('int64')
        return tf.convert_to_tensor(x, dtype=dtype)

def as_float_tensor(x):
    return as_tensor(x, tf.float32)

def as_double_tensor(x):
    return as_tensor(x, tf.float64)

def as_integer_tensor(x):
    return as_tensor(x, tf.int32)

def as_complex_tensor(x):
    return as_tensor(x, tf.complex64)

def is_empty(tensor):
    if not is_tensor(tensor):
        tensor = as_tensor(tensor)
    return tf.equal(tf.size(tensor), 0)

def as_scalar(x):
    if (len(tf.shape(x))):
        x = tf.squeeze(x)
        try:
            tf.assert_rank(x, 0)
        except:
            raise ValueError("Argument `x` must be of rank <= 1")
    return x

def is_scalar(x):
    if is_tensor(x):
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

def is_scalar_tensor(x, raise_err=False):
    if is_tensor(x):
        return x.ndim == 0
    if raise_err:
        raise ValueError("`x` is not a tensor")
    return False

def first(x):
    if is_scalar(x):
        return x
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[0] * len(x.shape)]

def last(x):
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[-1] * len(x.shape)]

def info(d, return_dict=False, print_=True):
    r"""Recursively grab shape, dtype, and size from (nested) dictionary of tensors"""
    info_ = {}
    for k,v in d.items():
        if isinstance(v, dict):
            info_.update(info(v))
        else:
            info_[k] = {
                'size': tf.size(np.asarray(v)).numpy(), 
                'shape' :np.asarray(v).shape, 
                'dtype': np.asarray(v).dtype.name
            }
            if print_:
                _v = np.asarray(v)
                print('key   -', k)
                print('dtype -', _v.dtype.name)
                print('size  -', tf.size(v).numpy())
                print('shape -', _v.shape)
                print()
    if return_dict:
        return info_

def maybe_list_up(x):
    if is_tensor(x):
        if len(tf.shape(x)) == 0:
            return [x]
    else:
        if len(np.asarray(x).shape) == 0:
            return [x]
    return x

def complex_range(x):
    R = tf.math.real(x)
    I = tf.math.imag(x)
    return { 
        'real': (tf.reduce_min(R).numpy(), tf.reduce_max(R).numpy()), 
        'imag': (tf.reduce_min(I).numpy(), tf.reduce_max(I).numpy())
    }

def tfrange(x):
    if is_complex_tensor(x):
        return complex_range(x)
    return (tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())

def normalize_range(x, lo=-1, hi=1):
    _min = tf.reduce_min(x)
    a = x - _min
    b = tf.reduce_max(x) - _min
    c = hi - lo
    return c * (a / b) + lo

def loadz(path, key='arr_0'):
    x = np.load(path, allow_pickle=True)
    if is_scalar(key):
        return x[key]
    return {k: v for k,v in x.items() if k in list(x.keys())}

def drop_none(x):
    if isinstance(x, dict):
        i = logical2idx(np.asarray(list(x.values())) == None)
        k = np.asarray(list(x))
        for key in k[i]:
            x.pop(key)
        return x
    if isinstance(x, list) or is_numpy(x):
        x = np.asarray(x)
        i = np.where(x != None)[0]
        return x[i]
    raise TypeError("Don't know how to process {} type".format(type(x)))

def plot_specs(spec, figsize=None, filename=None, title='', 
               xlab='time', ylab='freq_bins', hspace=0.5):
    shp = np.asarray(spec).shape
    if len(shp) < 2:
        raise ValueError("`spec` must be at least 2D")
    fig = plt.figure(figsize=(8,8))
    for i in range(shp[0]):
        fig.add_subplot(shp[0], 1, i+1)
        plt.imshow(
            spec[i], aspect='auto', interpolation='nearest', 
            cmap=plt.get_cmap('jet'), origin='lower')
        cbar = plt.colorbar()
        cbar.set_label('Amplitude (dB)')
    fig.subplots_adjust(hspace=hspace)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_spec(gram, tVec, fVec,
              use_pow=True,
              norm_db=True,
              vmin=-100,
              vmax=0,
              title='Traditional Spectrogram',
              save=True,
              outpath='spec.png'):
    plt.clf()
    fig1 = plt.figure(1)
    if use_pow:
        gram = np.power(np.abs(gram),2)
    if norm_db:
        gram = 10*np.log10((gram/np.max(gram)) + 1e-16)
    spec = plt.pcolormesh(tVec, fVec, gram, vmin=vmin, vmax=vmax)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Power (dB)')
    plt.title(title)
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_cep(spCep, nfft, tVec, tnum=100, 
             quefrencyUpsampleFactor=1, save=True, outpath='cep.png'):
    fig2 = plt.figure(2)
    spec = plt.pcolormesh(tVec,np.arange(0,tnum) / (
        quefrencyUpsampleFactor*nfft),10*np.log10(np.real(spCep)))
    plt.ylabel('Quefrency (sec)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Amplitude (dB)')
    plt.title('Traditional Cepstrogram')
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_ar_power(ar1, tVec, fVec, eps=1e-16, order=None, save=True, outpath='arpwr.png'):
    fig3 = plt.figure(3)
    arPwr = np.power(np.abs(ar1),2) 
    arPwrNrmDb = 10*np.log10((arPwr/np.max(arPwr)) + eps)
    spec = plt.pcolormesh(tVec,fVec,arPwrNrmDb,vmin=-100,vmax=0)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Power (dB)')
    plt.title('ARgram: Power, L = ' + str(order))
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_phase(ph, tVec, fVec, unwrap=False, order=None, save=True, outpath='arphase.png'):
    if unwrap:
        ph = np.unwrap(ph)
    if ph.shape[0] == tVec.shape:
        ph = ph.T # transpose
    fig4 = plt.figure(4)
    spec = plt.pcolormesh(tVec,fVec,ph)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Phase (deg)')
    plt.title('ARgram: Phase, L = ' + str(order))
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_group_delay(gd, tVec, fVec, fs=64000, vmin=0, vmax=0.005, 
                     order=None, save=True, outpath='group_delay.png'):
    fig5 = plt.figure(5)
    spec = plt.pcolormesh(tVec,fVec,gd/fs,vmin=vmin,vmax=vmax)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Group Delay (sec)')
    plt.title('ARgram: Group Delay, L = ' + str(order))
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_ar_cep(arCep, tVec, nfft, tnum=100, order=None, 
                quefrencyUpsampleFactor=1, save=True, outpath='arcep.png'):
    fig6 = plt.figure(6)
    if tVec.shape == arCep.shape[0]:
        arCep = arCep.T # transpose
    spec = plt.pcolormesh(
        tVec,
        np.arange(0,tnum)/(quefrencyUpsampleFactor*nfft),
        10*np.log10(np.real(arCep)))
    plt.ylabel('Quefrency (sec)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Amplitude (dB)')
    plt.title('ARgram: Cepstrum, L = ' + str(order))
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_ar_coeff(arCoeffHist, tVec, order=None, save=True, outpath='arcoeff.png'):
    if order is None:
        order = arCoeffHist.shape[-1] - 1
    fig7 = plt.figure(7)
    for ind in range(1, order):
        plt.plot(tVec, np.abs(arCoeffHist[:,ind]))
    plt.xlim(np.min(tVec),np.max(tVec))
    plt.title('AR Coefficent Amplitudes, L = ' + str(order))
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_noise_var(arNoiseVarHist, tVec, order=None, save=True, outpath='noisevar.png'):
    fig8 = plt.figure(8)
    plt.plot(tVec,arNoiseVarHist)
    plt.xlim(np.min(tVec),np.max(tVec))
    plt.title('AR Model Noise Variance, L = ' + str(order))
    plt.xlabel('Time (sec)')
    plt.ylabel('Power')
    if save:
        plt.savefig(outpath)
    else:
        plt.show()

def plot_all_grams(grams_dict, fs, tnum, plot_figures=True, save_figures=False, figdir='./'):
    tVec = grams_dict['tVec']
    fVec = grams_dict['fVec']
    # tnum = grams_dict['tnum']
    # fnum = grams_dict['fnum']
    sp = grams_dict['sp']
    ar = grams_dict['ar']
    ar1 = grams_dict['ar1']
    gd = grams_dict['group_delay']
    nfft = grams_dict['nfft']
    spCep = grams_dict['spCep']
    arCep = grams_dict['arCep']
    ARmodelOrder = grams_dict['ARmodelOrder']
    arNoiseVarHist = grams_dict['arNoiseVarHist']
    arCoeffHist = grams_dict['arCoeffHist']
    quefrencyUpsampleFactor = grams_dict['quefrencyUpsampleFactor']

    #Displays
    print('Plotting...')

    fig1 = plt.figure(1)
    spPwr = np.power(np.abs(sp),2) 
    spPwrNrmDb = 10*np.log10((spPwr/np.max(spPwr)) + 1e-16)
    spec = plt.pcolormesh(tVec,fVec,spPwrNrmDb,vmin=-100,vmax=0)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Power (dB)')
    plt.title('Traditional Spectrogram')

    fig2 = plt.figure(2)
    spec = plt.pcolormesh(
        tVec,
        np.arange(0,tnum)/(quefrencyUpsampleFactor*nfft),
        10*np.log10(np.real(spCep)))
    plt.ylabel('Quefrency (sec)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Amplitude (dB)')
    plt.title('Traditional Cepstrogram')

    fig3 = plt.figure(3)
    arPwr = np.power(np.abs(ar1),2) 
    arPwrNrmDb = 10*np.log10((arPwr/np.max(arPwr)) + 1e-16)
    spec = plt.pcolormesh(tVec,fVec,arPwrNrmDb,vmin=-100,vmax=0)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Power (dB)')
    plt.title('ARgram: Power, L = ' + str(ARmodelOrder))

    fig4 = plt.figure(4)
    spec = plt.pcolormesh(tVec,fVec,(np.angle(ar))*(180/np.pi))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Phase (deg)')
    plt.title('ARgram: Phase, L = ' + str(ARmodelOrder))

    fig5 = plt.figure(5)
    spec = plt.pcolormesh(tVec,fVec,np.unwrap(np.angle(ar))*(180/np.pi))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Unwrapped Phase (deg)')
    plt.title('ARgram: Unwrapped Phase, L = ' + str(ARmodelOrder))

    fig6 = plt.figure(6)
    spec = plt.pcolormesh(tVec,fVec,gd/fs,vmin=0,vmax=0.005)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Group Delay (sec)')
    plt.title('ARgram: Group Delay, L = ' + str(ARmodelOrder))

    fig7 = plt.figure(7)
    spec = plt.pcolormesh(
        tVec,
        np.arange(0,tnum)/(quefrencyUpsampleFactor*nfft),
        10*np.log10(np.real(arCep)))
    plt.ylabel('Quefrency (sec)')
    plt.xlabel('Time (sec)')
    cbar = plt.colorbar(spec)
    cbar.ax.set_ylabel('Amplitude (dB)')
    plt.title('ARgram: Cepstrum, L = ' + str(ARmodelOrder))

    fig8 = plt.figure(8)
    for ind in range(ARmodelOrder):
        plt.plot(tVec,np.abs(arCoeffHist[:,ind+1]))
    plt.xlim(np.min(tVec),np.max(tVec))
    plt.title('AR Coefficent Amplitudes, L = ' + str(ARmodelOrder))
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')

    fig9 = plt.figure(9)
    plt.plot(tVec,arNoiseVarHist)
    plt.xlim(np.min(tVec),np.max(tVec))
    plt.title('AR Model Noise Variance, L = ' + str(ARmodelOrder))
    plt.xlabel('Time (sec)')
    plt.ylabel('Power')

    if save_figures:
        print('Saving figures...')
        fig1.savefig(figdir + 'spectrogram_snr.png')
        fig2.savefig(figdir + 'spectrogram_cepstrogram.png')
        fig3.savefig(figdir + 'argram_snr.png')
        fig4.savefig(figdir + 'argram_phase.png')
        fig5.savefig(figdir + 'argram_unwrapped_phase.png')
        fig6.savefig(figdir + 'argram_group_delay.png')
        fig7.savefig(figdir + 'argram_cepstrogram.png')
        fig8.savefig(figdir + 'argram_coeffs.png')
        fig9.savefig(figdir + 'argram_noisevar.png')

    if plot_figures:
        plt.show()

from warnings import warn
def plot_metrics(history,
                 show=False,
                 save_png=True,
                 xlab=None, ylab=None,
                 xticks=None, xtick_labels=None,
                 outpath='training_curves_' + timestamp()):
    sns.set()
    plt.clf()
    plt.cla()

    keys = list(history.history)
    epochs = range(
        min(list(map(lambda x: len(x[1]), history.history.items())))
    ) 

    ax = plt.subplot(211)
    if 'acc' in keys:
        acc = history.history['acc']
    elif 'accuracy' in keys:
        acc = history.history['accuracy']
    elif 'train_acc' in keys:
        acc = history.history['train_acc']
    elif 'train_accuracy' in keys:
        acc = history.history['train_accuracy']
    elif 'sparse_categorical_accuracy' in keys:
        acc = history.history['sparse_categorical_accuracy']
    elif 'categorical_accuracy' in keys:
        acc = history.history['categorical_accuracy']
    else:
        raise ValueError("Training accuracy not found")


    plt.plot(epochs, acc, color='green', 
        marker='+', linestyle='dashed', label='Training accuracy'
    )

    if 'val_acc' in keys:
        val_acc = history.history['val_acc']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_accuracy' in keys:
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_sparse_categorical_accuracy' in keys:
        val_acc = history.history['val_sparse_categorical_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    elif 'val_categorical_accuracy' in keys:
        val_acc = history.history['val_categorical_accuracy']
        plt.plot(epochs, val_acc, color='blue', 
            marker='o', linestyle='dashed', label='Validation accuracy'
        )
    else:
        warn("Validation accuracy not found... skpping")

    if 'test_acc' in keys:
        test_acc = history.history['test_acc']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    elif 'test_accuracy' in keys:
        test_acc = history.history['test_accuracy']
        plt.plot(epochs, test_acc, color='red', 
            marker='x', linestyle='dashed', label='Test Accuracy'
        )
    else:
        warn("Test accuracy not found... skipping")


    plt.title('Training, validation and test accuracy')
    plt.legend()

    ax2 = plt.subplot(212)
    if 'loss' in keys:
        loss = history.history['loss']
    elif 'train_loss' in keys:
        loss = history.history['train_loss']
    else:
        raise ValueError("Training loss not found")

    plt.plot(epochs, loss, color='green', 
        marker='+', linestyle='dashed', label='Training Loss'
    )

    if 'val_loss' in keys:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, color='blue', 
            marker='o', linestyle='dashed', label='Validation Loss'
        )    
    elif 'validation_loss' in keys:
        val_loss = history.history['validation_loss']
        plt.plot(epochs, val_loss, color='blue', 
            marker='o', linestyle='dashed', label='Validation Loss'
        )
    else:
        warn("Validation loss not found... skipping")

    if 'test_loss' in keys:
        test_loss = history.history['test_loss']
        plt.plot(epochs, test_loss, color='red', marker='x', label='Test Loss')

    plt.title('Training, validation, and test loss')
    plt.legend()

    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax2.set_xticklabels(xtick_labels)

    plt.tight_layout()

    if save_png:
        plt.savefig(outpath)
    if show:
        plt.show()



def plot_mfcc(mfcc_db, vmin=None, outpath=None, title='MFCC Plot'):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    ax.set_title(title)
    ax.set_ylabel('MFCC Bin')
    ax.set_xlabel('Frame Number')
    if vmin is None:
        vmin = tf.reduce_min(mfcc_db)
    plt.imshow(
        mfcc_db, aspect='auto', interpolation='nearest', 
        cmap=plt.get_cmap('jet'), origin='lower', vmin=vmin, vmax=0
    )
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (dB)')
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.close()


def noNaNlossfn(fn):
    def noNaN(x, y):
        return fn(x, y+1e-8)
    return noNaN

def tf_counts(arr, x):
    arr = tf.constant(arr)
    return tf.where(arr == x).shape[0]

def counts(arr, x):
    arr = np.asarray(arr)
    return len(np.where(arr == x)[0])

def min_dataset():
    """Create a dataset with the smallest files available (usually for testing)"""
    from tf_dataset import signal_dataset
    from pandas import DataFrame
    MIN_FILE_A = '/array1/data/front_row_data/training/2019_08_01_east_node/1564664317.869_1564664325.071_24346_cargo.wav' # 460928
    MIN_FILE_B = '/array1/data/front_row_data/training/2018_12_01_east_node/1543669894.988_1543669907.948_24346_passenger.wav' # 829440 
    MIN_FILE_C = '/array1/data/front_row_data/training/2018_11_01_east_node/1541088020.705_1541088038.838_24346_towing.wav' # 1160512
    MIN_FILE_D = '/array1/data/front_row_data/training/2019_06_15_east_node/1560653959.922_1560653971.645_24346_tanker.wav' # 750272
    MIN_FILE_E = '/array1/data/front_row_data/training/2018_11_01_east_node/1541091481.426_1541091487.826_24346_cargo.wav' # 409600

    MIN_FILES = [MIN_FILE_A, MIN_FILE_B, MIN_FILE_C, MIN_FILE_D]
    df = DataFrame(
        {
            'filepath': MIN_FILES, 
            'class': ['cargo', 'passenger', 'towing', 'tanker'], 
            'target': [0, 1, 2, 3]
        }, 
        index=list(range(len(MIN_FILES)))
    )
    return signal_dataset(df, use_soundfile=True)
    
if False:
    from fr_train import *
    ds, vds = training_datasets(FLAGS)
    for x,y in ds: break
    model = load_and_build_model(FLAGS, x.shape, y.shape)
    h = scripts.train_wrapper.fit(model, ds, vds.take(5), epochs=5, steps_per_epoch=1)
    plot_metrics(h)

