import json
import os
import time

import h5py
import numpy as np

_STEP_NAME_API = 'Step index API'
vlen_bytes = h5py.special_dtype(vlen=bytes)

# HDF5 Enum datatypes matching official Labber SDK representation
interface_enum = h5py.special_dtype(enum=(np.int16, {'GPIB': 0, 'None': 7, 'Other': 6, 'PXI': 3, 'Serial': 4, 'TCPIP': 1, 'USB': 2, 'VISA': 5}))
startup_enum = h5py.special_dtype(enum=(np.int16, {'Do nothing': 2, 'Get config': 1, 'Set config': 0}))
range_type_enum = h5py.special_dtype(enum=(np.int16, {'Center - Span': 2, 'Single': 0, 'Start - Stop': 1}))
step_type_enum = h5py.special_dtype(enum=(np.int16, {'Fixed # of pts': 1, 'Fixed step': 0}))
interp_enum = h5py.special_dtype(enum=(np.int16, {'Linear': 0, 'Log': 1, 'Log, #/decade': 2, 'Lorentzian': 3}))
step_unit_enum = h5py.special_dtype(enum=(np.int16, {'Instrument': 0, 'Physical': 1}))
after_last_enum = h5py.special_dtype(enum=(np.int16, {'Goto first point': 0, 'Goto value...': 2, 'Stay at final': 1}))
sweep_mode_enum = h5py.special_dtype(enum=(np.int16, {'Between points': 1, 'Continuous': 2, 'Off': 0}))

def _get_database_folder():
    appdata = os.environ.get('APPDATA')
    if appdata:
        prefs_path = os.path.join(appdata, 'Labber', 'LabberPrefs.json')
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
                    db_folder = prefs.get('Database folder')
                    if db_folder:
                        return db_folder
            except Exception:
                pass
    return os.path.expanduser('~/Labber/Data')

def _create_log_path(sLogName, dateObj=None, logger_mode=False, bConfigFile=False, bCreatePath=True):
    if dateObj is None:
        import datetime
        dateObj = datetime.datetime.today()
    
    preferences_db = _get_database_folder()
    
    sYear = dateObj.strftime('%Y')
    sMonth = dateObj.strftime('%m')
    sDay = dateObj.strftime('Data_%m%d')
    
    sPath = os.path.join(preferences_db, sYear, sMonth, sDay)
    if bCreatePath:
        os.makedirs(sPath, exist_ok=True)
        
    return os.path.join(sPath, sLogName + '.hdf5')

def getTraceDict(y, x=None, x0=None, x1=None, dx=None):
    y = np.asarray(y)
    d = {'y': y}
    if x is not None:
        d['x'] = np.asarray(x)
        d['t0'] = 0.0
        d['dt'] = 1.0
    else:
        t0 = 0.0 if x0 is None else x0
        if dx is None:
            if x1 is not None and len(y) > 1:
                dt = (x1 - t0) / (len(y) - 1)
            else:
                dt = 1.0
        else:
            dt = dx
        d['t0'] = t0
        d['dt'] = dt
    return d

def _to_bytes(s):
    if isinstance(s, str):
        return s.encode('utf-8')
    return s

def _to_str(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s

class LogFile(object):
    def __init__(self, file_name):
        # Handle splitext logic like the SDK: append .hdf5 if not present
        base, ext = os.path.splitext(file_name)
        if ext.lower() != '.hdf5':
            file_name = base + '.hdf5'
        self.file_name = os.path.abspath(file_name)

    def getFilePath(self):
        return self.file_name

    def setComment(self, comment):
        with h5py.File(self.file_name, 'r+') as f:
            f.attrs['comment'] = _to_bytes(comment)

    def getComment(self):
        with h5py.File(self.file_name, 'r') as f:
            return _to_str(f.attrs.get('comment', b''))

    def setProject(self, project):
        with h5py.File(self.file_name, 'r+') as f:
            if 'Tags' in f:
                f['Tags'].attrs['Project'] = np.array([_to_bytes(project)], dtype=object)

    def getProject(self):
        with h5py.File(self.file_name, 'r') as f:
            if 'Tags' in f:
                proj = f['Tags'].attrs.get('Project', [b''])
                if len(proj) > 0:
                    return _to_str(proj[0])
            return ''

    def setUser(self, user):
        with h5py.File(self.file_name, 'r+') as f:
            if 'Tags' in f:
                f['Tags'].attrs['User'] = np.array([_to_bytes(user)], dtype=object)

    def getUser(self):
        with h5py.File(self.file_name, 'r') as f:
            if 'Tags' in f:
                usr = f['Tags'].attrs.get('User', [b''])
                if len(usr) > 0:
                    return _to_str(usr[0])
            return ''

    def setTags(self, tags):
        if tags is None:
            tags = []
        elif isinstance(tags, (str, bytes)):
            tags = [tags]

        with h5py.File(self.file_name, 'r+') as f:
            if 'Tags' in f:
                if len(tags) == 0:
                    f['Tags'].attrs['Tags'] = np.array([])
                else:
                    f['Tags'].attrs['Tags'] = np.array([_to_bytes(t) for t in tags], dtype=object)

    def getTags(self):
        with h5py.File(self.file_name, 'r') as f:
            if 'Tags' in f:
                tags = f['Tags'].attrs.get('Tags', [])
                return [_to_str(t) for t in tags]
            return []

    def getNumberOfEntries(self):
        # We need to return how many outer step entries have been written
        # In a completed or in-progress file, this corresponds to the size of Data/Time stamp
        # (or Log_N/Data/Time stamp).
        with h5py.File(self.file_name, 'r') as f:
            # Find the latest Log group or use root
            grp = f
            latest_grp_name = ''
            for key in sorted(f.keys()):
                if key.startswith('Log_'):
                    latest_grp_name = key
            if latest_grp_name:
                grp = f[latest_grp_name]
            
            if 'Data/Time stamp' in grp:
                return grp['Data/Time stamp'].shape[0]
            elif 'Traces/Time stamp' in grp:
                return grp['Traces/Time stamp'].shape[0]
            return 0

    def getStepChannels(self):
        with h5py.File(self.file_name, 'r') as f:
            # Step config keys or Channels entry where type matches
            channels = []
            if 'Channels' in f:
                # Type of Channels: name, instrument, quantity, unitPhys ...
                for row in f['Channels']:
                    name = _to_str(row['name'])
                    # If this channel is in Step list, it's a step channel
                    is_step = False
                    if 'Step list' in f:
                        for step_row in f['Step list']:
                            if _to_str(step_row['channel_name']) == name:
                                is_step = True
                                break
                    if is_step:
                        # Reconstruct values from Step config
                        values = np.array([])
                        cfg_path = f'Step config/{name}/Step items'
                        if cfg_path in f:
                            step_item = f[cfg_path][0]
                            # range_type: 1 (STARTSTOP), 0 (SINGLE)
                            # step_type: 1 (N_PTS)
                            range_type = step_item['range_type']
                            if range_type == 1:
                                values = np.linspace(step_item['start'], step_item['stop'], step_item['n_pts'])
                            elif range_type == 0:
                                values = np.array([step_item['single']])
                        
                        channels.append({
                            'name': name,
                            'unit': _to_str(row['unitPhys']),
                            'values': values,
                            'complex': False, # step channels are always real
                            'vector': False   # step channels are always scalar
                        })
            return channels

    def getLogChannels(self):
        with h5py.File(self.file_name, 'r') as f:
            channels = []
            if 'Channels' in f:
                # All channels that are in Log list
                log_names = set()
                if 'Log list' in f:
                    for log_row in f['Log list']:
                        log_names.add(_to_str(log_row['channel_name']))
                
                for row in f['Channels']:
                    name = _to_str(row['name'])
                    if name in log_names:
                        # Check if it is a vector channel
                        is_vector = False
                        is_complex = False
                        if 'Traces' in f and name in f['Traces']:
                            is_vector = True
                            is_complex = bool(f['Traces'][name].attrs.get('complex', False))
                        else:
                            # Check if it's complex from Instrument config or Data/Channel names
                            # If it's a scalar complex, its instrument config value would be complex
                            # or in Data/Channel names we have 'Real'/'Imaginary'
                            if 'Data/Channel names' in f:
                                # Count occurrences
                                infos = [(_to_str(r['name']), _to_str(r['info'])) for r in f['Data/Channel names']]
                                name_infos = [info for name_str, info in infos if name_str == name]
                                if 'Real' in name_infos or 'Imaginary' in name_infos:
                                    is_complex = True
                        
                        channels.append({
                            'name': name,
                            'unit': _to_str(row['unitPhys']),
                            'complex': is_complex,
                            'vector': is_vector
                        })
            return channels

    def getData(self, channel_name):
        with h5py.File(self.file_name, 'r') as f:
            # Check the latest group or root
            grp = f
            latest_grp_name = ''
            for key in sorted(f.keys()):
                if key.startswith('Log_'):
                    latest_grp_name = key
            if latest_grp_name:
                grp = f[latest_grp_name]

            # 1. Check if it's in Data/Channel names
            if 'Data/Channel names' in grp and 'Data/Data' in grp:
                chns = [(_to_str(row['name']), _to_str(row['info'])) for row in grp['Data/Channel names']]
                ch_indices = [idx for idx, (name, info) in enumerate(chns) if name == channel_name]
                if len(ch_indices) > 0:
                    ds = grp['Data/Data']
                    # ds shape: (dim1, num_channels, M)
                    # We want to return a 2D array of shape (M, dim1)
                    if len(ch_indices) == 1:
                        # Real/scalar channel
                        return ds[:, ch_indices[0], :].T
                    elif len(ch_indices) == 2:
                        # Complex scalar channel (stored as Real/Imaginary)
                        # Ensure real is first
                        real_idx = ch_indices[0] if chns[ch_indices[0]][1] == 'Real' else ch_indices[1]
                        imag_idx = ch_indices[1] if chns[ch_indices[1]][1] == 'Imaginary' else ch_indices[0]
                        return ds[:, real_idx, :].T + 1j * ds[:, imag_idx, :].T

            # 2. Check if it's in Traces
            if 'Traces' in grp and channel_name in grp['Traces']:
                ds = grp['Traces/' + channel_name]
                # ds shape: (trace_len, C, M)
                # We want to return a 2D array of shape (M, trace_len)
                trace_len = grp['Traces/' + channel_name + '_N'][0]
                is_complex = bool(ds.attrs.get('complex', False))
                C = ds.shape[1]
                
                if is_complex:
                    # Components 0: Real, 1: Imaginary
                    return ds[:trace_len, 0, :].T + 1j * ds[:trace_len, 1, :].T
                else:
                    if C == 1:
                        # Component 0: Y
                        return ds[:trace_len, 0, :].T
                    elif C == 2:
                        # Component 0: X, 1: Y
                        return ds[:trace_len, 1, :].T
                    
            raise ValueError(f"Channel {channel_name} not found in log file.")

    def getTraceXY(self, channel_name, entry=0):
        with h5py.File(self.file_name, 'r') as f:
            # Check the latest group or root
            grp = f
            latest_grp_name = ''
            for key in sorted(f.keys()):
                if key.startswith('Log_'):
                    latest_grp_name = key
            if latest_grp_name:
                grp = f[latest_grp_name]

            if 'Traces' in grp and channel_name in grp['Traces']:
                ds = grp['Traces/' + channel_name]
                trace_len = grp['Traces/' + channel_name + '_N'][0]
                is_complex = bool(ds.attrs.get('complex', False))
                C = ds.shape[1]
                col = entry
                
                # Check custom X by checking x_name / x_unit or the component size
                # Wait, if C == 2 and complex is False: custom X
                # If C == 3 and complex is True: custom X
                has_custom_x = (C == 2 and not is_complex) or (C == 3 and is_complex)
                
                # Reconstruct Y
                if is_complex:
                    y = ds[:trace_len, 0, col] + 1j * ds[:trace_len, 1, col]
                else:
                    if C == 1:
                        y = ds[:trace_len, 0, col]
                    elif C == 2:
                        y = ds[:trace_len, 1, col]
                
                # Reconstruct X
                if has_custom_x:
                    if is_complex:
                        x = ds[:trace_len, 2, col]
                    else:
                        x = ds[:trace_len, 0, col]
                else:
                    # Reconstruct from t0dt
                    t0dt_path = f'Traces/{channel_name}_t0dt'
                    if t0dt_path in grp:
                        t0 = grp[t0dt_path][0, 0]
                        dt = grp[t0dt_path][0, 1]
                    else:
                        t0, dt = 0.0, 1.0
                    x = t0 + np.arange(trace_len) * dt
                    
                return x, y
            
            raise ValueError(f"Channel {channel_name} not found in traces.")

    def addEntry(self, data):
        # We need to open the file, find the current write group, resize the datasets,
        # compute multi-indices, and write the data.
        with h5py.File(self.file_name, 'r+') as f:
            # Read step configuration to calculate step dims
            step_dims = list(f.attrs['Step dimensions'])
            if len(step_dims) == 0:
                step_dims = [1]
            dim1 = step_dims[0]
            M = np.prod(step_dims[1:]) if len(step_dims) > 1 else 1
            
            # Find the active group
            latest_grp_name = ''
            for key in sorted(f.keys()):
                if key.startswith('Log_'):
                    latest_grp_name = key
            
            grp = f
            if latest_grp_name:
                grp = f[latest_grp_name]
                
            # If the group already has M entries in its Data/Time stamp (or Traces/Time stamp):
            col = 0
            if 'Data/Time stamp' in grp:
                col = grp['Data/Time stamp'].shape[0]
            elif 'Traces/Time stamp' in grp:
                col = grp['Traces/Time stamp'].shape[0]
                
            if col >= M:
                # We must create a new group Log_N
                if latest_grp_name:
                    N = int(latest_grp_name.split('_')[1]) + 1
                else:
                    N = 2
                
                new_grp_name = f'Log_{N}'
                new_grp = f.create_group(new_grp_name)
                
                # Clone metadata from root
                for attr_key, attr_val in f.attrs.items():
                    new_grp.attrs[attr_key] = attr_val
                
                for key in ['Channels', 'Instrument config', 'Instruments', 'Log list', 'Step config', 'Step list', 'Tags']:
                    if key in f:
                        f.copy(key, f'{new_grp_name}/{key}')
                
                grp = new_grp
                col = 0
            
            # Now, we are ready to write to grp
            # Initialize Data group if not present
            if 'Data' not in grp:
                # We need to know which channels are scalar
                # Let's count them
                chn_names_list = []
                # First, all step channels (in order of step_channels)
                # Read step channels from Channels dataset that are in Step list
                step_names = []
                if 'Step list' in f:
                    for step_row in f['Step list']:
                        step_names.append(_to_str(step_row['channel_name']))
                
                # Read all channels
                log_names = []
                if 'Log list' in f:
                    for log_row in f['Log list']:
                        log_names.append(_to_str(log_row['channel_name']))
                
                # Channel configs
                channels_dict = {}
                if 'Channels' in f:
                    for row in f['Channels']:
                        name = _to_str(row['name'])
                        channels_dict[name] = {
                            'complex': False, # Default
                            'vector': False
                        }
                # Check if they are vector (if they are in Traces or configured so)
                # But wait, during initialization of data group, we only need to map Channels
                # that are scalar.
                # How does Labber identify complex scalar log channels?
                # A log channel is complex if configured so. We can scan the data passed to addEntry
                # or read from Instrument config defaults.
                # Let's check the keys in data passed. If any log channel key in data has a complex value
                # (np.iscomplexobj or isinstance(val, complex)), we treat it as complex.
                # Actually, let's look at the channels in Channels:
                # For each step name: it goes to Data/Channel names as (name, '')
                # For each log name: if it is not a vector channel (i.e. not in Traces and not having vector=True),
                # it goes to Data/Channel names.
                # Let's build the list of scalar channels:
                scalar_chns = []
                for name in step_names:
                    scalar_chns.append((name, ''))
                
                # Vector channels are those that are in Traces or have vector=True in data
                # Since we don't know yet if the user passes vector or complex, let's scan 'data'
                # to see if the log channels passed are complex or vector.
                for name in log_names:
                    val = data.get(name)
                    is_vector = False
                    is_complex = False
                    if val is not None:
                        # Check if dict (from getTraceDict)
                        if isinstance(val, dict) and ('y' in val):
                            is_vector = True
                            if np.iscomplexobj(val['y']):
                                is_complex = True
                        elif isinstance(val, (list, np.ndarray)) and not isinstance(val, (str, bytes)):
                            # It is a vector or a scalar array matching dim1.
                            # If it matches dim1, it is a scalar log channel (unless dim1 is 1, in which case
                            # it could be a vector of size 1? Yes, but usually if step_channels is empty or dim1=1
                            # and all are vector, bAllVector was True, so Step index API is the only step channel,
                            # and the log channel is vector.
                            # Wait, we can distinguish vector log channel by checking if the log channel
                            # was configured with x_name / x_unit or vector=True in createLogFile_ForData.
                            # We can find this out by looking at Instrument config for this log channel!
                            # In Instrument config, vector channels have attributes ___<name>___x_name!
                            pass
                    
                    # Read from Instrument config
                    inst_grp_path = 'Instrument config/Generic - GPIB: , Log channels at localhost'
                    if inst_grp_path in grp:
                        inst_grp = grp[inst_grp_path]
                        if f'___{name}___x_name' in inst_grp.attrs:
                            is_vector = True
                        # Check complex from default value of the attribute
                        if name in inst_grp.attrs:
                            default_val = inst_grp.attrs[name]
                            if isinstance(default_val, (complex, np.complexfloating)) or np.iscomplexobj(default_val):
                                is_complex = True
                    
                    if not is_vector:
                        if is_complex:
                            scalar_chns.append((name, 'Real'))
                            scalar_chns.append((name, 'Imaginary'))
                        else:
                            scalar_chns.append((name, ''))
                
                # Create Data group and datasets
                data_grp = grp.create_group('Data')
                data_grp.attrs['Completed'] = False
                data_grp.attrs['Step dimensions'] = step_dims
                data_grp.attrs['Step index'] = np.arange(len(step_dims), dtype=int)
                data_grp.attrs['Fixed step index'] = np.array([], dtype=int)
                data_grp.attrs['Fixed step values'] = np.array([], dtype=float)
                if len(step_dims) > 1:
                    data_grp.attrs['Entries, last trace'] = dim1
                
                # Channel names
                dtype_chn_names = np.dtype([('name', vlen_bytes), ('info', vlen_bytes)])
                chn_names_arr = np.array([(_to_bytes(n), _to_bytes(i)) for n, i in scalar_chns], dtype=dtype_chn_names)
                data_grp.create_dataset('Channel names', data=chn_names_arr)
                
                # Data/Data
                num_channels = len(scalar_chns)
                data_grp.create_dataset('Data', shape=(dim1, num_channels, 0), maxshape=(dim1, num_channels, M), chunks=(dim1, num_channels, M), dtype='f8')
                
                # Data/Time stamp
                data_grp.create_dataset('Time stamp', shape=(0,), maxshape=(M,), chunks=(M,), dtype='f8')
            
            # Now Data group exists
            data_grp = grp['Data']
            ds_data = data_grp['Data']
            ds_time = data_grp['Time stamp']
            
            # Resize
            ds_data.resize((dim1, ds_data.shape[1], col + 1))
            ds_time.resize((col + 1,))
            
            # Timestamp
            curr_timestamp = time.time()
            # Labber time stamp is offset from some epoch or local time?
            # In our sample, timestamp was around 3.1559 or 0.02. This is actually a relative elapsed time!
            # Wait, does the official SDK write elapsed time?
            # Yes! creation_time is a float timestamp, and Time stamp is the elapsed time since creation_time!
            # Let's verify: elapsed = curr_timestamp - f.attrs['creation_time'].
            creation_time = f.attrs.get('creation_time', curr_timestamp)
            elapsed_time = curr_timestamp - creation_time
            ds_time[col] = elapsed_time
            
            # Compute step channel values for this column
            # multi-index of outer loops
            outer_dims = step_dims[1:]
            if len(outer_dims) > 0:
                multi_idx = np.unravel_index(col, outer_dims)
            else:
                multi_idx = ()
            
            # Let's map channel names in Data/Channel names to their indices
            chns = [(_to_str(row['name']), _to_str(row['info'])) for row in data_grp['Channel names']]
            
            # 1. Step channel values
            # Read step channels names from Step list
            step_names = []
            if 'Step list' in f:
                for step_row in f['Step list']:
                    step_names.append(_to_str(step_row['channel_name']))
            
            # Reconstruct and write step channel values
            for idx, name in enumerate(step_names):
                # Reconstruct entire values array
                values = np.array([])
                cfg_path = f'Step config/{name}/Step items'
                if cfg_path in f:
                    step_item = f[cfg_path][0]
                    range_type = step_item['range_type']
                    if name == _STEP_NAME_API:
                        values = np.array([1.0])
                    elif range_type == 1:
                        values = np.linspace(step_item['start'], step_item['stop'], step_item['n_pts'])
                    elif range_type == 0:
                        values = np.array([step_item['single']])
                
                # Write to ds_data
                ch_idx = [i for i, (n, info) in enumerate(chns) if n == name][0]
                if idx == 0:
                    # Fastest loop
                    ds_data[:, ch_idx, col] = values
                else:
                    # Outer loop
                    val = values[multi_idx[idx - 1]]
                    ds_data[:, ch_idx, col] = val
                    
            # 2. Scalar log channels values from data dict
            for name, info in chns:
                if name in step_names:
                    continue
                # It is a scalar log channel
                val = data.get(name)
                if val is not None:
                    # Handle complex vs real
                    if info == '':
                        # Real scalar log channel
                        val_arr = np.asarray(val)
                        if val_arr.ndim == 0:
                            # Scalar, repeat dim1 times
                            ds_data[:, chns.index((name, '')), col] = float(val)
                        else:
                            ds_data[:, chns.index((name, '')), col] = val_arr
                    elif info == 'Real':
                        val_arr = np.asarray(val)
                        real_idx = chns.index((name, 'Real'))
                        imag_idx = chns.index((name, 'Imaginary'))
                        if val_arr.ndim == 0:
                            ds_data[:, real_idx, col] = np.real(val)
                            ds_data[:, imag_idx, col] = np.imag(val)
                        else:
                            ds_data[:, real_idx, col] = np.real(val_arr)
                            ds_data[:, imag_idx, col] = np.imag(val_arr)
            
            # 3. Vector log channels values
            # Read vector log names
            log_names = []
            if 'Log list' in f:
                for log_row in f['Log list']:
                    log_names.append(_to_str(log_row['channel_name']))
            
            vector_names = []
            inst_grp_path = 'Instrument config/Generic - GPIB: , Log channels at localhost'
            if inst_grp_path in grp:
                inst_grp = grp[inst_grp_path]
                for name in log_names:
                    if f'___{name}___x_name' in inst_grp.attrs:
                        vector_names.append(name)
            
            # Write vector log channels
            for name in vector_names:
                val = data.get(name)
                if val is not None:
                    # Reconstruct trace dict if not already
                    if not isinstance(val, dict) or 'y' not in val:
                        val = getTraceDict(val)
                    
                    y_data = np.asarray(val['y'])
                    trace_len = len(y_data)
                    t0 = val.get('t0', 0.0)
                    dt = val.get('dt', 1.0)
                    x_data = val.get('x', None)
                    is_complex = np.iscomplexobj(y_data)
                    
                    # Component count
                    if not is_complex and x_data is None:
                        C = 1
                    elif (not is_complex and x_data is not None) or (is_complex and x_data is None):
                        C = 2
                    else:
                        C = 3
                    
                    # Ensure Traces group exists
                    if 'Traces' not in grp:
                        grp.create_group('Traces')
                    traces_grp = grp['Traces']
                    
                    # Ensure dataset exists
                    ds_trace_path = name
                    if ds_trace_path not in traces_grp:
                        # Create trace datasets
                        traces_grp.create_dataset(name, shape=(trace_len, C, 0), maxshape=(None, C, M), chunks=(trace_len, C, max(1, M)), dtype='f8')
                        traces_grp.create_dataset(name + '_N', shape=(1,), dtype='i4')
                        traces_grp.create_dataset(name + '_t0dt', shape=(1, 2), dtype='f8')
                        
                        # Set attributes on trace dataset
                        ds_trace = traces_grp[name]
                        ds_trace.attrs['complex'] = is_complex
                        
                        # x_name and x_unit from Instrument config
                        x_name = _to_str(inst_grp.attrs.get(f'___{name}___x_name', b'Index'))
                        x_unit = _to_str(inst_grp.attrs.get(f'___{name}___x_unit', b''))
                        ds_trace.attrs['x, name'] = _to_bytes(x_name)
                        ds_trace.attrs['x, unit'] = _to_bytes(x_unit)
                    
                    ds_trace = traces_grp[name]
                    ds_N = traces_grp[name + '_N']
                    ds_t0dt = traces_grp[name + '_t0dt']
                    
                    # Resize along trace_len if needed
                    current_trace_len = ds_trace.shape[0]
                    if trace_len > current_trace_len:
                        ds_trace.resize((trace_len, C, ds_trace.shape[2]))
                    
                    # Resize along columns
                    ds_trace.resize((ds_trace.shape[0], C, col + 1))
                    
                    # Write trace data
                    if not is_complex and x_data is None:
                        ds_trace[:trace_len, 0, col] = y_data
                    elif not is_complex and x_data is not None:
                        ds_trace[:trace_len, 0, col] = x_data
                        ds_trace[:trace_len, 1, col] = y_data
                    elif is_complex and x_data is None:
                        ds_trace[:trace_len, 0, col] = np.real(y_data)
                        ds_trace[:trace_len, 1, col] = np.imag(y_data)
                    else:
                        ds_trace[:trace_len, 0, col] = np.real(y_data)
                        ds_trace[:trace_len, 1, col] = np.imag(y_data)
                        ds_trace[:trace_len, 2, col] = x_data
                        
                    # Write N
                    ds_N[0] = trace_len
                    # Write t0dt
                    if x_data is not None:
                        ds_t0dt[0, 0] = 0.0
                        ds_t0dt[0, 1] = 0.0
                    else:
                        ds_t0dt[0, 0] = t0
                        ds_t0dt[0, 1] = dt
            
            # Manage Traces/Time stamp if any traces are written
            if 'Traces' in grp:
                traces_grp = grp['Traces']
                if 'Time stamp' not in traces_grp:
                    traces_grp.create_dataset('Time stamp', shape=(0,), maxshape=(M,), chunks=(max(1, M),), dtype='f8')
                ds_trace_time = traces_grp['Time stamp']
                ds_trace_time.resize((col + 1,))
                ds_trace_time[col] = elapsed_time
            
            # Check completion
            if col == M - 1:
                if 'Data' in grp:
                    grp['Data'].attrs['Completed'] = True
                
                # Update time_per_point at root level on completion
                time_stamps = ds_time[:] if 'Data' in grp else grp['Traces/Time stamp'][:]
                if len(time_stamps) > 0:
                    time_sweep = np.diff(np.r_[0.0, time_stamps])
                    median_dt = np.median(time_sweep)
                    f.attrs['time_per_point'] = float(median_dt / dim1)

def createLogFile_ForData(name, log_channels, step_channels=[], use_database=True):
    # Resolve file path
    if use_database:
        resolved_path = _create_log_path(name)
    else:
        base, ext = os.path.splitext(name)
        if ext.lower() != '.hdf5':
            name = base + '.hdf5'
        resolved_path = os.path.abspath(name)
        parent_dir = os.path.dirname(resolved_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
    # Check if all log channels are vector
    # Default is vector=True if not specified
    for ch in log_channels:
        if 'vector' not in ch:
            ch['vector'] = True
            
    bAllVector = all([ch.get('vector', True) for ch in log_channels])
    
    # Prepend Step index API if all are vector
    actual_step_channels = list(step_channels)
    if bAllVector:
        step_index_api_ch = {
            'name': _STEP_NAME_API,
            'values': np.array([1.0])
        }
        actual_step_channels = [step_index_api_ch] + actual_step_channels
        
    # Create HDF5 file
    with h5py.File(resolved_path, 'w') as f:
        # 1. Root attributes
        step_dims = [len(ch['values']) for ch in actual_step_channels]
        f.attrs['Step dimensions'] = step_dims
        f.attrs['arm_trig_mode'] = False
        f.attrs['comment'] = b''
        f.attrs['creation_time'] = time.time()
        f.attrs['hardware_loop'] = False
        f.attrs['log_name'] = _to_bytes(os.path.splitext(os.path.basename(resolved_path))[0])
        f.attrs['log_parallel'] = True
        f.attrs['logger_mode'] = False
        f.attrs['time_per_point'] = 0.0
        f.attrs['trig_channel'] = b''
        f.attrs['version'] = b'1.8.6'
        f.attrs['wait_between'] = 0.01
        
        # 2. Tags group
        tags_grp = f.create_group('Tags')
        tags_grp.attrs['Project'] = np.array([b''], dtype=object)
        tags_grp.attrs['Tags'] = np.array([])
        tags_grp.attrs['User'] = np.array([b''], dtype=object)
        
        # 3. Settings & Instrument config group
        f.create_group('Settings')
        inst_cfg_grp = f.create_group('Instrument config')
        
        # 4. Instruments dataset
        dtype_instruments = np.dtype([
            ('hardware', vlen_bytes), ('version', vlen_bytes), ('id', vlen_bytes), ('model', vlen_bytes), ('name', vlen_bytes),
            ('interface', interface_enum), ('address', vlen_bytes), ('server', vlen_bytes), ('startup', startup_enum),
            ('lock', '?'), ('show_advanced', '?'), ('Timeout', '<f8'), ('Term. character', vlen_bytes),
            ('Send end on write', '?'), ('Lock VISA resource', '?'), ('Suppress end bit termination on read', '?'),
            ('Use specific TCP port', '?'), ('TCP port', '<f8'), ('Use VICP protocol', '?'),
            ('Baud rate', '<f8'), ('Data bits', '<f8'), ('Stop bits', '<f8'), ('Parity', vlen_bytes),
            ('GPIB board number', '<f8'), ('Send GPIB go to local at close', '?'), ('PXI chassis', '<f8'),
            ('Run in 32-bit mode', '?')
        ])
        
        step_inst_id = b'Generic - GPIB: , Step channels at localhost'
        log_inst_id = b'Generic - GPIB: , Log channels at localhost'
        
        inst_data = [
            (b'Generic', b'1.0', step_inst_id, b'', b'Step channels', 0, b'', b'', 0, False, False, 10.0, b'Auto', True, False, False, False, 0.0, False, 9600.0, 8.0, 1.0, b'No parity', 0.0, False, 1.0, False),
            (b'Generic', b'1.0', log_inst_id, b'', b'Log channels', 0, b'', b'', 0, False, False, 10.0, b'Auto', True, False, False, False, 0.0, False, 9600.0, 8.0, 1.0, b'No parity', 0.0, False, 1.0, False)
        ]
        f.create_dataset('Instruments', data=np.array(inst_data, dtype=dtype_instruments))
        
        # 5. Channels dataset
        dtype_channels = np.dtype([
            ('name', vlen_bytes), ('instrument', vlen_bytes), ('quantity', vlen_bytes), ('unitPhys', vlen_bytes), ('unitInstr', vlen_bytes),
            ('gain', '<f8'), ('offset', '<f8'), ('amp', '<f8'), ('highLim', '<f8'), ('lowLim', '<f8'),
            ('outputChannel', vlen_bytes), ('limit_action', vlen_bytes), ('limit_run_script', '?'), ('limit_script', vlen_bytes),
            ('use_log_interval', '?'), ('log_interval', '<f8'), ('limit_run_always', '?')
        ])
        
        chn_data = []
        for ch in actual_step_channels:
            name_bytes = _to_bytes(ch['name'])
            unit_bytes = _to_bytes(ch.get('unit', ''))
            chn_data.append((name_bytes, step_inst_id, name_bytes, unit_bytes, unit_bytes, 1.0, 0.0, 1.0, float('inf'), float('-inf'), b'', b'Nothing', False, b'', False, 1.0, False))
            
        for ch in log_channels:
            name_bytes = _to_bytes(ch['name'])
            unit_bytes = _to_bytes(ch.get('unit', ''))
            is_vector = ch.get('vector', True)
            high_lim = 0.0 if is_vector else float('inf')
            low_lim = 0.0 if is_vector else float('-inf')
            chn_data.append((name_bytes, log_inst_id, name_bytes, unit_bytes, unit_bytes, 1.0, 0.0, 1.0, high_lim, low_lim, b'', b'Nothing', False, b'', False, 1.0, False))
            
        f.create_dataset('Channels', data=np.array(chn_data, dtype=dtype_channels))
        
        # 6. Log list dataset
        dtype_log_list = np.dtype([('channel_name', vlen_bytes)])
        log_list_data = [(_to_bytes(ch['name']),) for ch in log_channels]
        f.create_dataset('Log list', data=np.array(log_list_data, dtype=dtype_log_list))
        
        # 7. Step list dataset
        dtype_step_list = np.dtype([
            ('channel_name', vlen_bytes), ('step_unit', step_unit_enum), ('wait_after', '<f8'), ('after_last', after_last_enum),
            ('final_value', '<f8'), ('use_relations', '?'), ('equation', vlen_bytes), ('show_advanced', '?'),
            ('sweep_mode', sweep_mode_enum), ('use_outside_sweep_rate', '?'), ('sweep_rate_outside', '<f8'),
            ('alternate_direction', '?')
        ])
        step_list_data = []
        for ch in actual_step_channels:
            step_list_data.append((_to_bytes(ch['name']), 0, float(ch.get('wait_after', 0.0)), 0, 0.0, False, b'x', False, 0, False, 0.0, False))
        f.create_dataset('Step list', data=np.array(step_list_data, dtype=dtype_step_list))
        
        # 8. Step config group & Instrument config sub-attributes
        step_cfg_grp = f.create_group('Step config')
        
        step_inst_cfg = inst_cfg_grp.create_group('Generic - GPIB: , Step channels at localhost')
        step_inst_cfg.attrs['Installed options'] = np.array([])
        for ch in actual_step_channels:
            name = ch['name']
            step_inst_cfg.attrs[name] = 0.0
            
            # Step config sub-groups
            single_step_grp = step_cfg_grp.create_group(name)
            
            # Optimizer group
            opt_grp = single_step_grp.create_group('Optimizer')
            opt_grp.attrs['Enabled'] = False
            
            vals = ch['values']
            if name == _STEP_NAME_API:
                vals = np.linspace(1.0, 2.0, 51)
                
            start = vals[0]
            stop = vals[-1]
            span = abs(stop - start)
            opt_grp.attrs['Initial step size'] = float(0.2 * span) if span > 0 else 1.0
            opt_grp.attrs['Max value'] = float(max(start, stop))
            opt_grp.attrs['Min value'] = float(min(start, stop))
            opt_grp.attrs['Precision'] = float(1e-4 * span) if span > 0 else 1e-4
            opt_grp.attrs['Start value'] = float(start)
            
            # Relation parameters dataset
            dtype_rel = np.dtype([('variable', vlen_bytes), ('channel_name', vlen_bytes), ('use_lookup', '?')])
            single_step_grp.create_dataset('Relation parameters', data=np.array([(b'x', b'Step values', False)], dtype=dtype_rel))
            
            # Step items dataset
            dtype_items = np.dtype([
                ('range_type', range_type_enum), ('step_type', step_type_enum), ('single', '<f8'), ('start', '<f8'),
                ('stop', '<f8'), ('center', '<f8'), ('span', '<f8'), ('step', '<f8'), ('n_pts', '<i4'),
                ('interp', interp_enum), ('sweep_rate', '<f8')
            ])
            if len(vals) > 1 and name != _STEP_NAME_API:
                item_data = (1, 1, start, start, stop, 0.0, 0.0, 0.0, len(vals), 0, 0.0)
            else:
                item_data = (0, 1, start, start, start + 1.0, 0.0, 0.0, 0.0, len(vals), 0, 0.0)
            single_step_grp.create_dataset('Step items', data=np.array([item_data], dtype=dtype_items))
            
        # 9. Instrument config for Log channels
        log_inst_cfg = inst_cfg_grp.create_group('Generic - GPIB: , Log channels at localhost')
        log_inst_cfg.attrs['Installed options'] = np.array([])
        for ch in log_channels:
            name = ch['name']
            is_vector = ch.get('vector', True)
            is_complex = ch.get('complex', False)
            if is_vector:
                x_name = ch.get('x_name', 'Time' if 'x_unit' in ch or 'x_name' in ch else 'Index')
                x_unit = ch.get('x_unit', 's' if 'x_unit' in ch or 'x_name' in ch else '')
                log_inst_cfg.attrs[f'___{name}___x_name'] = _to_bytes(x_name)
                log_inst_cfg.attrs[f'___{name}___x_unit'] = _to_bytes(x_unit)
                log_inst_cfg.attrs[name] = np.array([], dtype=float)
            else:
                log_inst_cfg.attrs[name] = 0.0j if is_complex else 0.0
                
    return LogFile(resolved_path)
