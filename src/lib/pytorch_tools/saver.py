import base64
import contextlib
import json
import os
import pickle

import torch

from ..logging import FileLogger, NullLogger, read_log_file

KWARGS_FILE = 'kwargs.json'
PARAMETERS_DIR = 'parameters'
METADATA_DIR = 'metadata'
LOGS_DIR = 'logs'
DEFAULT_PARAMETER_FILE = 'main'
DEFAULT_LOG_FILE = 'main.log'
DEFAULT_METADATA_NAME = 'main'

def construct_saver(model_constructor, directory_name, **kwargs):
    model = model_constructor(**kwargs)
    return construct_saver_from_model(model, directory_name, **kwargs)

def construct_saver_from_model(model, directory_name, **kwargs):
    if directory_name is None:
        return NullModelSaver(model, kwargs, {})
    else:
        return ModelSaver(
            model=model,
            kwargs=kwargs,
            directory_name=directory_name,
            created_output_dir=False,
            saved_kwargs=False,
            created_param_dir=False,
            created_metadata_dir=False,
            created_logs_dir=False,
            metadata_cache={},
            read_directory_name=directory_name
        )

def read_kwargs(directory_name):
    kwargs_path = os.path.join(directory_name, KWARGS_FILE)
    with open(kwargs_path) as fin:
        return json.load(fin)

def save_kwargs(directory_name, kwargs):
    kwargs_path = os.path.join(directory_name, KWARGS_FILE)
    with open(kwargs_path, 'w') as fout:
        write_json(fout, kwargs)

def read_saver(model_constructor, directory_name,
        parameter_file=DEFAULT_PARAMETER_FILE, device=None):
    kwargs = read_kwargs(directory_name)
    model = model_constructor(**kwargs)
    created_param_dir = False
    if parameter_file is not None:
        parameter_path = os.path.join(
            directory_name, PARAMETERS_DIR, parameter_file + '.pt')
        load_kwargs = {}
        if device is not None:
            load_kwargs['map_location'] = device
        model.load_state_dict(torch.load(parameter_path, **load_kwargs))
        created_param_dir = True
    if device is not None and device.type == 'cuda':
        model.to(device)
    return ModelSaver(
        model=model,
        kwargs=kwargs,
        directory_name=directory_name,
        created_output_dir=True,
        saved_kwargs=True,
        created_param_dir=created_param_dir,
        created_metadata_dir=False,
        created_logs_dir=False,
        metadata_cache={},
        read_directory_name=directory_name
    )

class ModelSaver:

    def __init__(self, model, kwargs, directory_name, created_output_dir,
            saved_kwargs, created_param_dir, created_metadata_dir,
            created_logs_dir, metadata_cache, read_directory_name):
        self.model = model
        self.kwargs = kwargs
        self.directory_name = directory_name
        self.created_output_dir = created_output_dir
        self.saved_kwargs = saved_kwargs
        self.created_param_dir = created_param_dir
        self.created_metadata_dir = created_metadata_dir
        self.created_logs_dir = created_logs_dir
        self.metadata_cache = metadata_cache
        self.read_directory_name = read_directory_name

    def save(self, file_name=DEFAULT_PARAMETER_FILE):
        self.ensure_output_dir_created()
        self.ensure_kwargs_file_written()
        self.ensure_param_dir_created()
        param_path = os.path.join(
            self.directory_name, PARAMETERS_DIR, file_name + '.pt')
        torch.save(self.model.state_dict(), param_path)

    def save_metadata(self, data, name=DEFAULT_METADATA_NAME):
        self.ensure_output_dir_created()
        file_name = os.path.join(self.directory_name, METADATA_DIR, name + '.json')
        self.ensure_metadata_dir_created()
        with open(file_name, 'w') as fout:
            write_json(fout, data)
        self.metadata_cache[name] = data

    def metadata(self, path, name=DEFAULT_METADATA_NAME):
        if name in self.metadata_cache:
            data = self.metadata_cache[name]
        else:
            file_name = os.path.join(self.read_directory_name, METADATA_DIR,
                name + '.json')
            with open(file_name) as fin:
                data = json.load(fin)
            self.metadata_cache[name] = data
        return path_lookup(data, path)

    def check_output(self):
        self.ensure_output_dir_created()

    @contextlib.contextmanager
    def logger(self, name=DEFAULT_LOG_FILE):
        self.ensure_output_dir_created()
        self.ensure_logs_dir_created()
        file_name = os.path.join(self.directory_name, LOGS_DIR, name)
        # If a directory is specified, open the log file and return a
        # logger object.
        # Attempt to open the log file in *exclusive* mode so the
        # operation will fail early if the log file already exists.
        with open(file_name, 'x') as fout:
            logger = FileLogger(fout)
            try:
                yield logger
            except KeyboardInterrupt:
                # Log an event if the program was interrupted.
                logger.log('keyboard_interrupt')
                raise
            except Exception as e:
                # Log other kinds of exceptions.
                try:
                    e_str = str(e)
                except Exception:
                    e_str = str(type(e))
                logger.log('exception', { 'exception' : e_str })
                raise

    def logs(self, name=DEFAULT_LOG_FILE):
        return read_logs(self.directory_name, name)

    def ensure_output_dir_created(self):
        if not self.created_output_dir:
            try:
                os.makedirs(self.directory_name)
            except FileExistsError:
                raise DirectoryExists('the directory %s already exists' % self.directory_name)
            self.created_output_dir = True

    def ensure_kwargs_file_written(self):
        if not self.saved_kwargs:
            save_kwargs(self.directory_name, self.kwargs)
            self.saved_kwargs = True

    def ensure_param_dir_created(self):
        if not self.created_param_dir:
            param_dir_path = os.path.join(self.directory_name, PARAMETERS_DIR)
            os.makedirs(param_dir_path, exist_ok=True)
            self.created_param_dir = True

    def ensure_metadata_dir_created(self):
        if not self.created_metadata_dir:
            metadata_dir_path = os.path.join(self.directory_name, METADATA_DIR)
            os.makedirs(metadata_dir_path, exist_ok=True)
            self.created_metadata_dir = True

    def ensure_logs_dir_created(self):
        if not self.created_logs_dir:
            logs_dir_path = os.path.join(self.directory_name, LOGS_DIR)
            os.makedirs(logs_dir_path, exist_ok=True)
            self.created_logs_dir = True

    def to_directory(self, directory_name):
        if directory_name is None:
            # TODO Allow this to use self.read_directory_name
            return NullModelSaver(
                model=self.model,
                kwargs=self.kwargs,
                metadata_cache=self.metadata_cache
            )
        else:
            return ModelSaver(
                model=self.model,
                kwargs=self.kwargs,
                directory_name=directory_name,
                created_output_dir=False,
                saved_kwargs=False,
                created_param_dir=False,
                created_metadata_dir=False,
                created_logs_dir=False,
                metadata_cache=self.metadata_cache,
                read_directory_name=self.read_directory_name
            )

def write_json(fout, data):
    json.dump(data, fout, indent=2, sort_keys=True)

def path_lookup(data, path):
    if isinstance(path, str):
        path = path.split('.')
    for key in path:
        data = data[key]
    return data

@contextlib.contextmanager
def read_logs(directory_name, name=DEFAULT_LOG_FILE):
    file_name = os.path.join(directory_name, LOGS_DIR, name)
    with open(file_name) as fin:
        yield read_log_file(fin)

class DirectoryExists(Exception):
    pass

def _serialize_pytorch_object(obj):
    class_name = type(obj).__name__
    state_dict = obj.state_dict()
    pickled_state_dict = pickle.dumps(state_dict)
    str_state_dict = base64.b64encode(pickled_state_dict).decode('ascii')
    return { 'class_name' : class_name, 'state_dict' : str_state_dict }

def _deserialize_pytorch_object(namespace, data, *args, **kwargs):
    Class = getattr(namespace, data['class_name'])
    obj = Class(*args, **kwargs)
    str_state_dict = data['state_dict']
    pickled_state_dict = base64.b64decode(str_state_dict.encode('ascii'))
    state_dict = pickle.loads(pickled_state_dict)
    obj.load_state_dict(state_dict)
    return obj

def serialize_optimizer(optimizer):
    return _serialize_pytorch_object(optimizer)

def deserialize_optimizer(data, parameters):
    return _deserialize_pytorch_object(torch.optim, data, parameters, lr=1.0)

def serialize_lr_scheduler(scheduler):
    return _serialize_pytorch_object(scheduler)

def deserialize_lr_scheduler(data, *args, **kwargs):
    return _deserialize_pytorch_object(
        torch.optim.lr_scheduler, data, *args, **kwargs)

class NullModelSaver:

    def __init__(self, model, kwargs, metadata_cache):
        self.model = model
        self.kwargs = kwargs
        self.metadata_cache = metadata_cache

    def save(self, file_name=None):
        pass

    def save_metadata(self, data, name=DEFAULT_METADATA_NAME):
        self.metadata_cache[name] = data

    def metadata(self, path, name=DEFAULT_METADATA_NAME):
        return path_lookup(self.metadata_cache[name], path)

    def check_output(self):
        pass

    @contextlib.contextmanager
    def logger(self, name=None):
        yield NullLogger()

    def to_directory(self, directory_name):
        if directory_name is None:
            return self
        else:
            # TODO
            raise NotImplementedError
