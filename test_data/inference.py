import json
from pathlib import Path
import subprocess


DEFAULT_CONFIG_PATH = 'default_test_model/config.json'
DEFAULT_CHECKPOINT_PATH = 'default_test_model/checkpoint.pth'


def change_arch_param(config_path, param_name, value):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['arch']['args'][param_name] = value
    with open(config_path, 'w') as f:
        json.dump(config, f)


out_dir = Path('test_data') / 'audio' / 'normal'
out_dir.mkdir(exist_ok=True, parents=True)
subprocess.run([
    'python', 'test.py',
    '-c', DEFAULT_CONFIG_PATH,
    '-r', DEFAULT_CHECKPOINT_PATH,
    '-o', str(out_dir),
])


for param_name in ['duration', 'pitch', 'energy']:
    for value in [0.8, 1.2]:
        change_arch_param(DEFAULT_CONFIG_PATH, param_name + '_control', value)

        out_dir = Path('test_data') / 'audio' / f'{param_name}{int(value * 100)}'
        out_dir.mkdir(exist_ok=True)
        subprocess.run([
            'python', 'test.py',
            '-c', DEFAULT_CONFIG_PATH,
            '-r', DEFAULT_CHECKPOINT_PATH,
            '-o', str(out_dir),
        ])

        change_arch_param(DEFAULT_CONFIG_PATH, param_name + '_control', 1.0)


for value in [0.8, 1.2]:
    for param_name in ['duration', 'pitch', 'energy']:
        change_arch_param(DEFAULT_CONFIG_PATH, param_name + '_control', value)

    out_dir = Path('test_data') / 'audio' / f'duration_pitch_energy{int(value * 100)}'
    out_dir.mkdir(exist_ok=True)
    subprocess.run([
        'python', 'test.py',
        '-c', DEFAULT_CONFIG_PATH,
        '-r', DEFAULT_CHECKPOINT_PATH,
        '-o', str(out_dir),
    ])

    for param_name in ['duration', 'pitch', 'energy']:
        change_arch_param(DEFAULT_CONFIG_PATH, param_name + '_control', 1.0)
