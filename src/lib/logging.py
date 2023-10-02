import datetime
import functools
import json

class Logger:

    def log(self, event_type, data=None):
        raise NotImplementedError

class NullLogger(Logger):

    def log(self, event_type, data=None):
        pass

class FileLogger(Logger):

    def __init__(self, file, flush=False, reopen=False):
        super().__init__()
        self.file = file
        self.flush = flush
        self.reopen = reopen

    def log(self, event_type, data=None):
        self.log_event(LogEvent(event_type, get_current_time(), data, None))

    def log_event(self, event):
        self.file.write(event.type)
        self.file.write(' ')
        self.file.write(str(event.timestamp.timestamp()))
        if event.has_data():
            self.file.write(' ')
            event.write_data(self.file)
        self.file.write('\n')
        if self.reopen:
            self.file.close()
            self.file = open(self.file.name, 'a')
        elif self.flush:
            self.file.flush()

def read_log_file(file):
    return map(parse_log_line, file)

class LogParseError(ValueError):
    pass

def parse_log_line(line):
    try:
        fields = line.split(' ', 2)
        try:
            event_type, timestamp_str = fields
        except ValueError:
            event_type, timestamp_str, data_str = fields
        else:
            data_str = None
        timestamp = parse_timestamp(float(timestamp_str))
    except ValueError:
        raise LogParseError(f'cannot parse log line {line!r}')
    return LogEvent(event_type, timestamp, None, data_str)

class LogEvent:

    def __init__(self, type, timestamp, data, data_str):
        self.type = type
        self.timestamp = timestamp
        self._data = data
        self._data_str = data_str

    @functools.cached_property
    def data(self):
        if self._data is not None:
            return self._data
        elif self._data_str is not None:
            return json.loads(self._data_str)
        else:
            return None

    def has_data(self):
        return self._data is not None or self._data_str is not None

    def write_data(self, fout):
        if self._data_str is not None:
            fout.write(self._data_str)
        elif self._data is not None:
            json.dump(self._data, fout, separators=(',', ':'), sort_keys=True)

def get_current_time():
    return datetime.datetime.now(datetime.timezone.utc)

def parse_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
