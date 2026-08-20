from comfy_execution.graph import ExecutionBlocker  # type: ignore
import folder_paths  # type: ignore
import torch  # type: ignore
import numpy as np
from PIL import Image, ImageColor
from math import ceil
from fractions import Fraction
import hashlib
import json
import os
import re
import shutil
import time
from types import SimpleNamespace

from ..collection.register_nodes import register_node
from ..shared.utils import ANY_TYPE
from ..shared.plot_data import (
    DimData,
    XYPlotQueueData,
    PlotConfigHFData,
    PlotConfigGridData,
    PlotVars,
)
from ..shared.pager import Pager
from ..shared.utils import pillow_to_tensor, tensor_to_pillow

# common tooltips
TOOLTIP_XY_PLOT_DATA = (
    'data sent by the queue, to indicate the cuttent state of processing'
)
TOOLTIP_COLOR = (
    "can be either a color name ('red'), or the RGB hex notation ('#b9b9b9', '#aaa')"
)
TOOLTIP_FONT = "font: either one of 'Roboto-Regular.ttf' / 'Roboto-Bold.ttf' / 'Roboto-Italic.ttf' / 'Roboto-BoldItalic.ttf',\nor the full path to a locally-installed TTF font"
TOOLTIP_FONT_SIZE = 'font size'
TOOLTIP_FONT_COLOR = 'font color, ' + TOOLTIP_COLOR
TOOLTIP_HF_PLACEHOLDERS = (
    'the following placeholders are accepted: {current_page}, {total_pages}'
)
TOOLTIP_PLOT_CONFIG_GRID = 'plot configuration for the grid'
_RESAMPLE_LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')


def _frame_rate(value, fallback=24.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(fallback)
    return value if value > 0 else float(fallback)


def _copy_audio(audio):
    if not isinstance(audio, dict):
        return audio
    waveform = audio.get('waveform')
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu()
    return {
        'waveform': waveform,
        'sample_rate': int(audio.get('sample_rate', 0)),
    }


def _video_record(value, fps=24.0):
    """Normalize VIDEO or IMAGE input to CPU frames plus optional audio/FPS."""
    audio = None
    frame_rate = fps
    if isinstance(value, torch.Tensor):
        frames = value
    elif hasattr(value, 'get_components'):
        try:
            components = value.get_components()
        except Exception as exc:
            raise RuntimeError(f'Could not read the input video components: {exc}') from exc
        frames = getattr(components, 'images', None)
        audio = getattr(components, 'audio', None)
        frame_rate = getattr(components, 'frame_rate', fps)
    else:
        raise TypeError(
            'XY Plot: Video nodes expect a VIDEO object or an IMAGE frame batch'
        )

    if not isinstance(frames, torch.Tensor):
        raise ValueError('The video decoder did not provide an IMAGE tensor of frames')
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(
            f'Video frames must have shape [frames, height, width, channels], got {tuple(frames.shape)}'
        )
    if frames.shape[0] < 1:
        raise ValueError('The video contains no frames')
    if frames.shape[-1] not in (3, 4):
        raise ValueError(
            f'Video frames must have 3 or 4 channels, got {frames.shape[-1]}'
        )
    if frames.shape[-1] == 4:
        # ComfyUI's video encoder currently expects RGB frames. The plot is
        # also composited onto a solid background, so dropping alpha here is
        # preferable to producing a VIDEO object that cannot be saved.
        frames = frames[..., :3]

    return SimpleNamespace(
        frames=frames.detach().cpu(),
        audio=_copy_audio(audio),
        frame_rate=_frame_rate(frame_rate, fps),
    )


def _video_from_record(record):
    try:
        from comfy_api.latest import InputImpl, Types  # type: ignore

        return InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=record.frames,
                audio=record.audio,
                frame_rate=Fraction(round(record.frame_rate * 1000), 1000),
            )
        )
    except ImportError as exc:
        raise RuntimeError(
            'This ComfyUI version does not expose the VIDEO API required by XY Plot: Video'
        ) from exc


def _background_rgb(value, fallback='black'):
    value = fallback if value in (None, '', 'transparent') else value
    try:
        rgb = ImageColor.getrgb(value)
    except (TypeError, ValueError):
        rgb = ImageColor.getrgb(fallback)
    return rgb[:3]


def _plot_data_to_dict(xy_plot_data):
    return {
        'index': xy_plot_data.index,
        'current_page': xy_plot_data.current_page,
        'total_pages': xy_plot_data.total_pages,
        'complete': xy_plot_data.complete,
        'dim1': {
            'index': xy_plot_data.dim1.index,
            'length': xy_plot_data.dim1.length,
            'value': xy_plot_data.dim1.value,
        },
        'dim2': {
            'index': xy_plot_data.dim2.index,
            'length': xy_plot_data.dim2.length,
            'value': xy_plot_data.dim2.value,
        },
    }


def _canonical_runtime_value(value):
    if isinstance(value, dict):
        return {
            key: _canonical_runtime_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_runtime_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_prompt_node(
    prompt,
    node_id,
    seen=None,
    cell_values=None,
    queue_index=None,
    stable_cell_values=True,
):
    if seen is None:
        seen = set()
    node_id = str(node_id)
    if node_id in seen:
        return ['cycle', node_id]
    node = prompt.get(node_id)
    if node is None:
        return ['missing', node_id]

    seen = seen | {node_id}

    def canonical_input(value):
        if (
            isinstance(value, list)
            and len(value) == 2
            and str(value[0]) in prompt
            and isinstance(value[1], int)
        ):
            source = prompt[str(value[0])]
            if (
                source.get('class_type') == 'XYPlotQueue'
                and cell_values is not None
                and value[1] in (1, 2)
            ):
                return [
                    'xy_cell_value',
                    value[1],
                    _canonical_runtime_value(cell_values[value[1] - 1]),
                ]
            return [
                'link',
                _canonical_prompt_node(
                    prompt,
                    value[0],
                    seen,
                    cell_values,
                    queue_index,
                    stable_cell_values,
                ),
                value[1],
            ]
        if isinstance(value, dict):
            return {key: canonical_input(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [canonical_input(item) for item in value]
        return value

    inputs = dict(node.get('inputs', {}))
    index = inputs.get('index', 0)
    if node.get('class_type') == 'XYPlotQueue':
        if cell_values is not None and stable_cell_values:
            # The selected dimension values identify a media cell. The
            # queue's mutable continuation index must not make the same cell
            # produce a different key on another run or page scan.
            inputs['index'] = 0
        elif queue_index is not None:
            # Compatibility lookup for caches created before cell-stable keys
            # used the queue position as part of their signature.
            inputs['index'] = queue_index
        elif isinstance(index, (int, float)) and index < 0:
            inputs['index'] = 0
    canonical_inputs = {}
    for key in sorted(inputs):
        if (
            node.get('class_type') == 'XYPlotQueue'
            and cell_values is not None
            and key in ('dim1', 'dim2')
        ):
            # A queue cell consumes one value from each dimension. Including
            # the complete LIST node here makes changing one prompt invalidate
            # every cell, even though the other cells do not use that prompt.
            dimension_index = 0 if key == 'dim1' else 1
            canonical_inputs[key] = [
                'xy_dimension_value',
                key,
                _canonical_runtime_value(cell_values[dimension_index]),
            ]
        else:
            canonical_inputs[key] = canonical_input(inputs[key])

    return [node.get('class_type'), canonical_inputs]


class _XYPlotImageCache:
    def __init__(
        self,
        prompt=None,
        unique_id=None,
        cache_key='',
        cell_values=None,
        input_name='image',
        queue_index=None,
        key=None,
    ):
        self.root = os.path.join(
            folder_paths.get_user_directory(), 'comfylab', 'xy_cache'
        )
        self.legacy_root = os.path.join(
            folder_paths.get_temp_directory(), 'comfylab_xy_cache'
        )
        if key is not None:
            self.key = key
            self.primary_key = key
            self.fallback_keys = ()
            self._migrate_legacy_entry()
            return
        node = prompt.get(str(unique_id), {})
        image_link = node.get('inputs', {}).get(input_name)
        if not (
            isinstance(image_link, list)
            and len(image_link) == 2
            and str(image_link[0]) in prompt
        ):
            self.key = None
            self.primary_key = None
            self.fallback_keys = ()
            return

        def make_key(
            selected_cell_values,
            selected_queue_index=None,
            stable_cell_values=True,
        ):
            signature = [
                _canonical_prompt_node(
                    prompt,
                    image_link[0],
                    cell_values=selected_cell_values,
                    queue_index=selected_queue_index,
                    stable_cell_values=stable_cell_values,
                ),
                image_link[1],
                cache_key,
            ]
            encoded = json.dumps(
                signature,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=False,
            ).encode('utf-8')
            return hashlib.sha256(encoded).hexdigest()

        self.key = make_key(cell_values)
        self.primary_key = self.key
        # Connecting XY plot data enables cell-stable keys. Existing caches
        # made before that connection used the complete queue signature, so
        # retain that key as a read-only compatibility fallback. The optional
        # metadata input must not force already-generated media to rerender.
        fallback_keys = []
        if cell_values is not None:
            # Compatibility with caches created by the first cell-aware
            # implementation, which still included the mutable queue index.
            fallback_keys.append(
                make_key(cell_values, stable_cell_values=False)
            )
            if queue_index is not None:
                fallback_keys.append(
                    make_key(
                        cell_values,
                        queue_index,
                        stable_cell_values=False,
                    )
                )
            # Compatibility with caches created before xy_plot_data was
            # connected to the cache node, which used a positional queue key.
            fallback_keys.append(make_key(None, queue_index))
        self.fallback_keys = tuple(
            key for key in fallback_keys if key is not None and key != self.key
        )
        self._migrate_legacy_entry()

    def _migrate_legacy_entry(self, key=None):
        key = self.key if key is None else key
        if key is None or not os.path.isdir(self.legacy_root):
            return
        manifest_path = os.path.join(self.root, key + '.json')
        if os.path.isfile(manifest_path):
            return
        names = [
            name
            for name in os.listdir(self.legacy_root)
            if name == key + '.json' or name.startswith(key + '.')
        ]
        if not names:
            return
        os.makedirs(self.root, exist_ok=True)
        for name in names:
            shutil.copy2(
                os.path.join(self.legacy_root, name),
                os.path.join(self.root, name),
            )

    @property
    def manifest_path(self):
        return os.path.join(self.root, self.key + '.json')

    def _candidate_keys(self):
        keys = [self.key]
        if self.primary_key not in keys:
            keys.append(self.primary_key)
        keys.extend(key for key in self.fallback_keys if key not in keys)
        return [key for key in keys if key is not None]

    def _remove_key(self, key):
        if key is None or not os.path.isdir(self.root):
            return
        for name in os.listdir(self.root):
            if name == key + '.json' or name.startswith(key + '.'):
                try:
                    os.remove(os.path.join(self.root, name))
                except FileNotFoundError:
                    pass

    def _load_frames(self, manifest):
        frames = []
        for index in range(manifest['frames']):
            path = os.path.join(self.root, f'{self.key}.{index}.png')
            with Image.open(path) as image:
                frames.append(pillow_to_tensor(image.convert(manifest['mode'])))
        return torch.cat(frames, dim=0)

    def load(self):
        manifest = self.load_manifest()
        if manifest is None:
            return None
        try:
            frames = self._load_frames(manifest)
            now = time.time()
            os.utime(self.manifest_path, (now, now))
            return frames
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            self.remove()
            return None

    def load_media(self, default_frame_rate=24.0):
        manifest = self.load_manifest()
        if manifest is None:
            return None
        try:
            frames = self._load_frames(manifest)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            self.remove()
            return None

        audio = None
        audio_info = manifest.get('audio')
        if isinstance(audio_info, dict):
            audio_file = audio_info.get('file')
            audio_path = os.path.join(self.root, audio_file) if audio_file else None
            if audio_path and os.path.isfile(audio_path):
                try:
                    waveform = torch.from_numpy(
                        np.load(audio_path, allow_pickle=False)
                    )
                    sample_rate = int(audio_info.get('sample_rate', 0))
                    if sample_rate > 0:
                        audio = {
                            'waveform': waveform,
                            'sample_rate': sample_rate,
                        }
                except (OSError, ValueError, EOFError):
                    # A missing/corrupt audio sidecar must not invalidate the
                    # cached video frames. The user can still render silently.
                    audio = None

        now = time.time()
        try:
            os.utime(self.manifest_path, (now, now))
        except OSError:
            pass
        return SimpleNamespace(
            frames=frames,
            audio=audio,
            frame_rate=_frame_rate(
                manifest.get('frame_rate', default_frame_rate), default_frame_rate
            ),
        )

    def load_manifest(self):
        for key in self._candidate_keys():
            self._migrate_legacy_entry(key)
            path = os.path.join(self.root, key + '.json')
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    manifest = json.load(file)
            except (OSError, ValueError, json.JSONDecodeError):
                self._remove_key(key)
                continue
            self.key = key
            return manifest
        return None

    def save(
        self,
        image,
        max_cache_mb,
        xy_plot_data=None,
        frame_rate=None,
        audio=None,
    ):
        if self.key is None:
            return
        os.makedirs(self.root, exist_ok=True)
        mode = 'RGBA' if image.shape[-1] == 4 else 'RGB'
        for index, frame in enumerate(image):
            array = (
                frame.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
            )
            Image.fromarray(array, mode=mode).save(
                os.path.join(self.root, f'{self.key}.{index}.png')
            )
        manifest = {'frames': image.shape[0], 'mode': mode}
        if frame_rate is not None:
            manifest['frame_rate'] = _frame_rate(frame_rate)
        if isinstance(audio, dict) and isinstance(audio.get('waveform'), torch.Tensor):
            sample_rate = int(audio.get('sample_rate', 0))
            if sample_rate > 0:
                audio_file = f'{self.key}.audio.npy'
                np.save(
                    os.path.join(self.root, audio_file),
                    audio['waveform'].detach().cpu().numpy(),
                )
                manifest['audio'] = {
                    'file': audio_file,
                    'sample_rate': sample_rate,
                }
        if xy_plot_data is not None:
            manifest['xy_plot_data'] = _plot_data_to_dict(xy_plot_data)
        with open(self.manifest_path, 'w', encoding='utf-8') as file:
            json.dump(manifest, file)
        self.prune(max_cache_mb)

    def save_plot_data(self, xy_plot_data):
        manifest = self.load_manifest()
        if manifest is None:
            return
        manifest['xy_plot_data'] = _plot_data_to_dict(xy_plot_data)
        with open(self.manifest_path, 'w', encoding='utf-8') as file:
            json.dump(manifest, file)

    def remove(self):
        for key in self._candidate_keys():
            self._remove_key(key)

    def prune(self, max_cache_mb):
        limit = max_cache_mb * 1024 * 1024
        manifests = []
        total = 0
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if os.path.isfile(path):
                total += os.path.getsize(path)
            if name.endswith('.json') and name != self.key + '.json':
                manifests.append((os.path.getmtime(path), name[:-5]))
        for _, key in sorted(manifests):
            if total <= limit:
                break
            for name in os.listdir(self.root):
                if name == key + '.json' or name.startswith(key + '.'):
                    path = os.path.join(self.root, name)
                    try:
                        total -= os.path.getsize(path)
                        os.remove(path)
                    except FileNotFoundError:
                        pass


def _cached_plot_cells(prompt, queue_id, dim1, dim2):
    total = len(dim1) * len(dim2)
    for node_id, node in prompt.items():
        cache_type = node.get('class_type')
        if cache_type not in ('XYPlotImageCache', 'XYPlotVideoCache'):
            continue
        input_name = 'video' if cache_type == 'XYPlotVideoCache' else 'image'
        inputs = node.get('inputs', {})
        cache_key = inputs.get('cache_key', '')
        if cache_type == 'XYPlotVideoCache':
            cache_key = f"{cache_key}|fps={_frame_rate(inputs.get('fps', 24.0))}"
        xy_link = inputs.get('xy_plot_data')
        if not (
            isinstance(xy_link, list)
            and len(xy_link) == 2
            and str(xy_link[0]) == str(queue_id)
            and inputs.get('cache_mode', 'use cache') == 'use cache'
        ):
            continue

        cells = []
        for index in range(total):
            dim1_index = int(index / len(dim2))
            dim2_index = index % len(dim2)
            cache = _XYPlotImageCache(
                prompt,
                node_id,
                cache_key,
                cell_values=(dim1[dim1_index], dim2[dim2_index]),
                input_name=input_name,
                queue_index=index,
            )
            manifest = cache.load_manifest()
            if manifest is None or (
                manifest.get('frames', 0) < 1
                or not all(
                    os.path.isfile(
                        os.path.join(cache.root, f"{cache.key}.{frame}.png")
                    )
                    for frame in range(manifest['frames'])
                )
            ):
                cells = []
                break

            # Cell metadata is derived from the current queue, not stored in
            # the media manifest. This permits identical videos to be shared
            # by multiple cells without one cell overwriting another cell's
            # row/column metadata.
            plot_data = _plot_data_to_dict(
                XYPlotQueueData(
                    index,
                    0,
                    1,
                    index == total - 1,
                    DimData(dim1_index, len(dim1), dim1[dim1_index]),
                    DimData(dim2_index, len(dim2), dim2[dim2_index]),
                )
            )
            cells.append({'cache_key': cache.key, 'plot_data': plot_data})
        if len(cells) == total:
            return cells
    return None


def _plot_data_from_dict(data):
    return XYPlotQueueData(
        data['index'],
        data['current_page'],
        data['total_pages'],
        data['complete'],
        DimData(**data['dim1']),
        DimData(**data['dim2']),
    )


def _xy_cell_coords(cell):
    match = re.fullmatch(r'([A-Za-z]+)([1-9][0-9]*)', cell.strip())
    if match is None:
        raise ValueError("XY Plot cell must look like 'A1', 'B3', or 'AA12'")
    col = 0
    for char in match.group(1).upper():
        col = col * 26 + ord(char) - ord('A') + 1
    return (col - 1, int(match.group(2)) - 1)


def _xy_column_label(index):
    label = ''
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord('A') + remainder) + label
    return label


@register_node('XY Plot: Queue', 'plot')
class XYPlotQueue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'dim1': (
                    'LIST',
                    {
                        'tooltip': 'list of values, processed by queue as the 1st dimension\ntip: associate slow operation, like loading checkpoint, to dim1, to ensure better performance'
                    },
                ),
            },
            'optional': {
                'dim2': (
                    'LIST',
                    {
                        'tooltip': 'optional: list of values, processed by queue as the 2nd dimension'
                    },
                ),
                'index': ('QUEUE_STATUS', {'tooltip': 'current queue status'}),
                'max_dim1_per_page': (
                    'INT',
                    {
                        'default': 0,
                        'min': 0,
                        'tooltip': 'optional: if > 0, max number of dim1 values per page',
                    },
                ),
                'max_dim2_per_page': (
                    'INT',
                    {
                        'default': 0,
                        'min': 0,
                        'tooltip': 'optional: if > 0, max number of dim2 values per page',
                    },
                ),
            },
            'hidden': {
                'prompt': 'PROMPT',
                'unique_id': 'UNIQUE_ID',
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('XY_PLOT_DATA', ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ('xy_plot_data', 'dim1_value', 'dim2_value')
    OUTPUT_TOOLTIPS = (
        TOOLTIP_XY_PLOT_DATA,
        'dim1 value (not typed)',
        'dim2 value (not typed)',
    )
    DESCRIPTION = 'Loop through all values of dim1, optionally combined with dim2 values, and send them to outputs.\nIMPORTANT: for a given dim1 value, all dim2 values are first iterated, before going to the next dim1 value.\nSo it is advised to associate slow operations (e.g. loading checkpoints) to dim1, to ensure better performance.'

    def run(
        self,
        dim1: list[str],
        max_dim1_per_page: int,
        max_dim2_per_page: int,
        index: int,
        dim2: list[str] = [''],
        prompt=None,
        unique_id=None,
    ):
        starting_plot = index < 0
        if index < 0:  # value has been reset (completion, interrupt, errors)
            index = 0

        if len(dim2) == 0:
            dim2 = ['']

        # to simplify code below
        size = (len(dim1), len(dim2))
        dim1_limited = max_dim1_per_page > 0 and max_dim1_per_page < size[0]
        dim2_limited = max_dim2_per_page > 0 and max_dim2_per_page < size[1]

        # calculate total
        total = size[0] * size[1]

        # check if we have finished
        complete = index == total - 1

        # max page dimensions, but not necessarily the current page ones: the last one may be smaller
        max_page_dims = (
            max_dim1_per_page if dim1_limited else size[0],
            max_dim2_per_page if dim2_limited else size[1],
        )

        # page index and total nb of pages
        total_pages = (
            ceil(size[0] / max_page_dims[0]),
            ceil(size[1] / max_page_dims[1]),
        )
        # compute the index and dimension of the current page for dim1 first
        current_page_index_dim1 = int(index / (size[1] * max_page_dims[0]))
        current_page_dims_dim1 = min(
            max_page_dims[0], size[0] - current_page_index_dim1 * max_page_dims[0]
        )

        # compute the index and dimension of the current page for dim2
        current_page = (
            current_page_index_dim1,
            int(
                (index - current_page_index_dim1 * size[1] * max_page_dims[0])
                / (current_page_dims_dim1 * max_page_dims[1])
            )
            % size[1],
        )
        current_page_dims = (
            current_page_dims_dim1,
            min(max_page_dims[1], size[1] - current_page[1] * max_page_dims[1]),
        )

        # sequential index and dim1/dim2 indexes in current page
        index_in_page_seq = (
            index
            - current_page[0] * size[1] * max_page_dims[0]
            - current_page[1] * current_page_dims[0] * max_page_dims[1]
        )
        index_in_page = (
            int(index_in_page_seq / current_page_dims[1]),
            int(index_in_page_seq % current_page_dims[1]),
        )

        # get value from each list, given the paged index
        values = (
            dim1[
                int(index_in_page_seq / current_page_dims[1])
                + current_page[0] * max_page_dims[0]
            ],
            dim2[
                int(index_in_page_seq % current_page_dims[1])
                + current_page[1] * max_page_dims[1]
            ],
        )

        # build data sent to XYPlotRender
        xy_plot_data = XYPlotQueueData(
            index_in_page_seq,
            current_page[0] * total_pages[1] + current_page[1],
            total_pages[0] * total_pages[1],
            complete,
            DimData(index_in_page[0], current_page_dims[0], values[0]),
            DimData(index_in_page[1], current_page_dims[1], values[1]),
        )

        if starting_plot and total_pages == (1, 1):
            cached_cells = _cached_plot_cells(prompt, unique_id, dim1, dim2)
            if cached_cells is not None:
                xy_plot_data.cached_cells = cached_cells
                return {
                    'result': (xy_plot_data, values[0], values[1]),
                    'ui': {'index': [total - 1], 'total': [total]},
                }

        return {
            'result': (xy_plot_data, values[0], values[1]),
            'ui': {'index': [index], 'total': [total]},
        }


@register_node('XY Plot: Image Cache', 'plot')
class XYPlotImageCache:
    def __init__(self):
        self.cached_image = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'image': ('IMAGE', {'lazy': True}),
                'cache_mode': (
                    ['use cache', 'refresh', 'bypass'],
                    {
                        'default': 'use cache',
                        'tooltip': 'use cache skips the image branch when the same generation inputs were cached. Persistent cache location: <ComfyUI user directory>/comfylab/xy_cache',
                    },
                ),
                'cache_key': (
                    'STRING',
                    {
                        'default': '',
                        'tooltip': 'optional namespace to keep similar plots in separate caches',
                    },
                ),
                'max_cache_mb': (
                    'INT',
                    {
                        'default': 2048,
                        'min': 64,
                        'max': 65536,
                        'tooltip': 'maximum persistent disk space used by all XY plot image caches',
                    },
                ),
            },
            'optional': {
                'xy_plot_data': (
                    'XY_PLOT_DATA',
                    {
                        'tooltip': 'connect XY Plot: Queue here to enable instant whole-plot rendering'
                    },
                ),
            },
            'hidden': {
                'prompt': 'PROMPT',
                'unique_id': 'UNIQUE_ID',
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    DESCRIPTION = 'Cache each generated plot image at <ComfyUI user directory>/comfylab/xy_cache. Place this node immediately before XY Plot: Render to re-render plot styling without sampling again, including after restarting ComfyUI.'

    def check_lazy_status(
        self,
        cache_mode,
        cache_key,
        max_cache_mb,
        image=None,
        xy_plot_data=None,
        prompt=None,
        unique_id=None,
    ):
        self.cached_image = None
        if image is not None or cache_mode == 'bypass':
            return []
        cell_values = (
            (xy_plot_data.dim1.value, xy_plot_data.dim2.value)
            if xy_plot_data is not None
            else None
        )
        cache = _XYPlotImageCache(
            prompt, unique_id, cache_key, cell_values=cell_values
        )
        if cache_mode == 'use cache':
            self.cached_image = cache.load()
            if self.cached_image is not None:
                return []
        return ['image']

    def run(
        self,
        cache_mode,
        cache_key,
        max_cache_mb,
        image=None,
        xy_plot_data=None,
        prompt=None,
        unique_id=None,
    ):
        if cache_mode == 'bypass':
            return (image,)

        cell_values = (
            (xy_plot_data.dim1.value, xy_plot_data.dim2.value)
            if xy_plot_data is not None
            else None
        )
        cache = _XYPlotImageCache(
            prompt, unique_id, cache_key, cell_values=cell_values
        )
        if cache_mode == 'use cache':
            cached_image = self.cached_image
            self.cached_image = None
            if cached_image is None:
                cached_image = cache.load()
            if cached_image is not None:
                if xy_plot_data is not None:
                    cache.save_plot_data(xy_plot_data)
                return (cached_image,)

        if image is None:
            raise RuntimeError('XY Plot image cache could not load or generate an image')
        cache.remove()
        cache.save(image, max_cache_mb, xy_plot_data)
        return (image,)


@register_node('XY Plot: Video Cache', 'plot')
class XYPlotVideoCache:
    def __init__(self):
        self.cached_video = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'video': (
                    ANY_TYPE,
                    {
                        'lazy': True,
                        'tooltip': 'VIDEO or an IMAGE batch containing video frames',
                    },
                ),
                'fps': (
                    'FLOAT',
                    {
                        'default': 24.0,
                        'min': 0.01,
                        'max': 1000.0,
                        'step': 0.01,
                        'tooltip': 'FPS used when the input is an IMAGE frame batch; VIDEO inputs keep their own FPS',
                    },
                ),
                'cache_mode': (
                    ['use cache', 'refresh', 'bypass'],
                    {
                        'default': 'use cache',
                        'tooltip': 'use cache skips the video branch when the same generation inputs were cached. Persistent cache location: <ComfyUI user directory>/comfylab/xy_cache',
                    },
                ),
                'cache_key': (
                    'STRING',
                    {
                        'default': '',
                        'tooltip': 'optional namespace to keep similar plots in separate caches',
                    },
                ),
                'max_cache_mb': (
                    'INT',
                    {
                        'default': 4096,
                        'min': 64,
                        'max': 65536,
                        'tooltip': 'maximum persistent disk space used by all XY plot caches',
                    },
                ),
            },
            'optional': {
                'xy_plot_data': (
                    'XY_PLOT_DATA',
                    {
                        'tooltip': 'connect XY Plot: Queue here to enable instant whole-plot rendering'
                    },
                ),
            },
            'hidden': {
                'prompt': 'PROMPT',
                'unique_id': 'UNIQUE_ID',
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('video',)
    DESCRIPTION = 'Cache each XY cell as video frames and optional audio at <ComfyUI user directory>/comfylab/xy_cache. Accepts VIDEO or decoded IMAGE frame batches.'

    def _cache(
        self,
        prompt,
        unique_id,
        cache_key,
        fps,
        xy_plot_data,
    ):
        cell_values = (
            (xy_plot_data.dim1.value, xy_plot_data.dim2.value)
            if xy_plot_data is not None
            else None
        )
        return _XYPlotImageCache(
            prompt,
            unique_id,
            f"{cache_key}|fps={_frame_rate(fps)}",
            cell_values=cell_values,
            input_name='video',
        )

    def check_lazy_status(
        self,
        fps,
        cache_mode,
        cache_key,
        max_cache_mb,
        video=None,
        xy_plot_data=None,
        prompt=None,
        unique_id=None,
    ):
        self.cached_video = None
        if video is not None or cache_mode == 'bypass':
            return []
        cache = self._cache(prompt, unique_id, cache_key, fps, xy_plot_data)
        if cache_mode == 'use cache':
            cached_record = cache.load_media(fps)
            if cached_record is not None:
                self.cached_video = _video_from_record(cached_record)
                return []
        return ['video']

    def run(
        self,
        fps,
        cache_mode,
        cache_key,
        max_cache_mb,
        video=None,
        xy_plot_data=None,
        prompt=None,
        unique_id=None,
    ):
        if cache_mode == 'bypass':
            if video is None:
                raise RuntimeError('XY Plot: Video Cache received no video or frame batch')
            return (_video_from_record(_video_record(video, fps)),)

        cache = self._cache(prompt, unique_id, cache_key, fps, xy_plot_data)
        if cache_mode == 'use cache':
            cached_video = self.cached_video
            self.cached_video = None
            if cached_video is None:
                cached_record = cache.load_media(fps)
                if cached_record is not None:
                    cached_video = _video_from_record(cached_record)
            if cached_video is not None:
                if xy_plot_data is not None:
                    cache.save_plot_data(xy_plot_data)
                return (cached_video,)

        if video is None:
            raise RuntimeError(
                'XY Plot: Video Cache could not load or generate a video cell'
            )
        record = _video_record(video, fps)
        cache.remove()
        cache.save(
            record.frames,
            max_cache_mb,
            xy_plot_data,
            frame_rate=record.frame_rate,
            audio=record.audio,
        )
        return (_video_from_record(record),)


@register_node('XY Plot: Select Cell', 'plot')
class XYPlotSelectCell:
    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'xy_plot_data': ('XY_PLOT_DATA', {'tooltip': TOOLTIP_XY_PLOT_DATA}),
                'image': ('IMAGE', {'lazy': True}),
                'cell': (
                    'STRING',
                    {
                        'default': 'A1',
                        'tooltip': "Excel-style visual grid position, such as 'A1' or 'C3'",
                    },
                ),
                'direction': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'label_on': 'dim1 as rows',
                        'label_off': 'dim1 as cols',
                        'tooltip': 'must match the direction selected on XY Plot: Render',
                    },
                ),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('selected_image',)
    OUTPUT_TOOLTIPS = ('image at the selected visual XY plot cell',)
    DESCRIPTION = "Select one XY plot image using an Excel-style cell such as A1 or C3. Connect the output to Preview Image or Save Image."

    def _matches(self, xy_plot_data, cell, direction):
        col, row = _xy_cell_coords(cell)
        max_cols = xy_plot_data.dim2.length if direction else xy_plot_data.dim1.length
        max_rows = xy_plot_data.dim1.length if direction else xy_plot_data.dim2.length
        if col >= max_cols or row >= max_rows:
            last_col = _xy_column_label(max_cols - 1)
            requested_col = _xy_column_label(col)
            example_col = requested_col if col < max_cols else 'A'
            raise ValueError(
                f"XY Plot cell {cell.upper()} does not exist. This plot has {max_cols} columns and {max_rows} rows, "
                f"so valid cells range from A1 to {last_col}{max_rows}. The number is the visual row position, "
                f"not an epoch value. If epoch {row + 1} is the second item in your epoch list, use "
                f"{example_col}2 instead."
            )
        dim1_index, dim2_index = (row, col) if direction else (col, row)
        return (
            xy_plot_data.dim1.index == dim1_index
            and xy_plot_data.dim2.index == dim2_index
        )

    def check_lazy_status(
        self, xy_plot_data, cell, direction, image=None
    ):
        if xy_plot_data.cached_cells is not None:
            return []
        if self._matches(xy_plot_data, cell, direction) and image is None:
            return ['image']
        return []

    def run(
        self, xy_plot_data, cell, direction, image=None
    ):
        if xy_plot_data.cached_cells is not None:
            for cached_cell in xy_plot_data.cached_cells:
                cell_data = _plot_data_from_dict(cached_cell['plot_data'])
                if self._matches(cell_data, cell, direction):
                    selected = _XYPlotImageCache(
                        key=cached_cell['cache_key']
                    ).load()
                    if selected is None:
                        raise RuntimeError(
                            'The selected cached XY Plot image is missing; run the plot again to rebuild it'
                        )
                    return (selected,)
            raise RuntimeError(f'XY Plot cell {cell.upper()} was not found in cache')

        if not self._matches(xy_plot_data, cell, direction):
            return (ExecutionBlocker(None),)
        return (image,)


@register_node('XY Plot: Render', 'plot')
class XYPlotRender:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'xy_plot_data': ('XY_PLOT_DATA', {'tooltip': TOOLTIP_XY_PLOT_DATA}),
                'image': ('IMAGE', {'lazy': True}),
                'dim1_header_format': (
                    'STRING',
                    {
                        'default': '{dim1}',
                        'tooltip': "template text to be displayed as dim1 header.\nthe '{dim1}' placeholder will be replaced by the current value.\nSafe expressions include {dim1[:20]}; string replacement is also supported.\nUse '\\n' for multiline text.",
                    },
                ),
                'dim2_header_format': (
                    'STRING',
                    {
                        'default': '{dim2}',
                        'tooltip': "template text to be displayed as dim2 header.\nthe '{dim2}' placeholder will be replaced by the current value.\nSafe expressions include {dim2[:20]}; string replacement is also supported.\nUse '\\n' for multiline text.",
                    },
                ),
                'direction': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'label_on': 'dim1 as rows',
                        'label_off': 'dim1 as cols',
                        'tooltip': 'display dim1 values as rows or columns',
                    },
                ),
            },
            'optional': {
                'plot_config_grid': (
                    'PLOT_CONFIG_GRID',
                    {'tooltip': 'optional: ' + TOOLTIP_PLOT_CONFIG_GRID},
                ),
                'plot_config_header': (
                    'PLOT_CONFIG_HF',
                    {'tooltip': 'optional: plot configuration for the page header'},
                ),
                'plot_config_footer': (
                    'PLOT_CONFIG_HF',
                    {'tooltip': 'optional: plot configuration for the page footer'},
                ),
                'group_dim2_headers': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': 'group Cartesian-product DIM2 columns by their first value',
                    },
                ),
                'dim2_group_header_format': (
                    'STRING',
                    {
                        'default': 'Prompt: {dim2_group:.60}…',
                        'tooltip': "group header template. Use '{dim2_group}' for the first Cartesian-product value",
                    },
                ),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('grid', 'image')
    OUTPUT_TOOLTIPS = (
        'grid page',
        'individual image, as received in input',
    )
    DESCRIPTION = 'Render the generated images as grids.\nOptional configuration is available to customize the look of the grid, and page header / footer.'

    # Pager object, to hold each individual images and build the grid
    pager = None

    def check_lazy_status(self, xy_plot_data, image=None, **kwargs):
        if xy_plot_data.cached_cells is not None:
            return []
        if image is None:
            return ['image']
        return []

    def run(
        self,
        xy_plot_data: XYPlotQueueData,
        image: torch.Tensor,
        dim1_header_format: str,
        dim2_header_format: str,
        direction: bool,
        plot_config_grid=PlotConfigGridData(),
        plot_config_header=None,
        plot_config_footer=None,
        group_dim2_headers=False,
        dim2_group_header_format='Prompt: {dim2_group:.60}…',
    ):
        if xy_plot_data.cached_cells is not None:
            first_data = _plot_data_from_dict(
                xy_plot_data.cached_cells[0]['plot_data']
            )
            self.pager = Pager(
                first_data,
                (dim1_header_format, dim2_header_format),
                direction,
                group_dim2_headers,
                dim2_group_header_format,
            )
            for cell in xy_plot_data.cached_cells:
                cell_data = _plot_data_from_dict(cell['plot_data'])
                image = _XYPlotImageCache(key=cell['cache_key']).load()
                if image is None:
                    raise RuntimeError(
                        'A cached XY Plot image disappeared; run the plot again to rebuild it'
                    )
                self.pager.add(cell_data, image)
            grid = self.pager.make_grid(
                PlotVars(1, 1),
                plot_config_grid,
                plot_config_header,
                plot_config_footer,
            )
            return (grid, image)

        if xy_plot_data.index == 0 or self.pager is None:
            self.pager = Pager(
                xy_plot_data,
                (dim1_header_format, dim2_header_format),
                direction,
                group_dim2_headers,
                dim2_group_header_format,
            )

        # add image to pager
        self.pager.add(xy_plot_data, image)

        # check if page is complete
        if not self.pager.complete:
            # block downstream nodes for grid output, just send the individual image
            return {'result': (ExecutionBlocker(None), image)}
        else:
            grid = self.pager.make_grid(
                PlotVars(xy_plot_data.current_page + 1, xy_plot_data.total_pages),
                plot_config_grid,
                plot_config_header,
                plot_config_footer,
            )
            # complete, send grid + individual image
            return (
                grid,
                image,
            )


def _video_cell_indices(cells, cell, direction):
    if not cells:
        raise ValueError('The XY plot has no cells from which to select audio')
    first_data = cells[0][0]
    col, row = _xy_cell_coords(cell)
    max_cols = first_data.dim2.length if direction else first_data.dim1.length
    max_rows = first_data.dim1.length if direction else first_data.dim2.length
    if col >= max_cols or row >= max_rows:
        last_col = _xy_column_label(max_cols - 1)
        raise ValueError(
            f'Audio cell {cell.upper()} does not exist. This plot has {max_cols} columns and {max_rows} rows, '
            f'so valid cells range from A1 to {last_col}{max_rows}'
        )
    dim1_index, dim2_index = (row, col) if direction else (col, row)
    for index, (plot_data, _) in enumerate(cells):
        if (
            plot_data.dim1.index == dim1_index
            and plot_data.dim2.index == dim2_index
        ):
            return index
    raise ValueError(f'Audio cell {cell.upper()} was not found in the current XY plot page')


def _video_frame_image(frame, size, resize_mode, pad_rgb):
    target_width, target_height = size
    image = tensor_to_pillow(frame).convert('RGB')
    if resize_mode == 'stretch':
        return image.resize((target_width, target_height), _RESAMPLE_LANCZOS)

    scale = min(target_width / image.width, target_height / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(resized_size, _RESAMPLE_LANCZOS)
    canvas = Image.new('RGB', (target_width, target_height), color=pad_rgb)
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return canvas


@register_node('XY Plot: Video Render', 'plot')
class XYPlotVideoRender:
    def __init__(self):
        self.cells = None
        self.page = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'xy_plot_data': ('XY_PLOT_DATA', {'tooltip': TOOLTIP_XY_PLOT_DATA}),
                'video': (
                    ANY_TYPE,
                    {
                        'lazy': True,
                        'tooltip': 'VIDEO or an IMAGE batch containing video frames',
                    },
                ),
                'dim1_header_format': (
                    'STRING',
                    {
                        'default': '{dim1}',
                        'tooltip': "template text for the dim1 header; use '{dim1}' and '\\n' for multiline text",
                    },
                ),
                'dim2_header_format': (
                    'STRING',
                    {
                        'default': '{dim2}',
                        'tooltip': "template text for the dim2 header; use '{dim2}' and '\\n' for multiline text",
                    },
                ),
                'direction': (
                    'BOOLEAN',
                    {
                        'default': True,
                        'label_on': 'dim1 as rows',
                        'label_off': 'dim1 as cols',
                        'tooltip': 'display dim1 values as rows or columns',
                    },
                ),
                'audio_mode': (
                    ['silent', 'first cell', 'first available', 'selected cell'],
                    {
                        'default': 'silent',
                        'tooltip': 'keep no audio, or keep one audio track from the plot; audio tracks are never mixed',
                    },
                ),
                'audio_cell': (
                    'STRING',
                    {
                        'default': 'A1',
                        'tooltip': "Excel-style cell used when audio mode is 'selected cell'",
                    },
                ),
                'temporal_padding': (
                    ['repeat last frame', 'black', 'white', 'plot background'],
                    {
                        'default': 'repeat last frame',
                        'tooltip': 'how to extend videos shorter than the longest cell video',
                    },
                ),
                'resolution_mode': (
                    ['largest', 'first cell'],
                    {
                        'default': 'largest',
                        'tooltip': 'choose the common cell resolution used by the plot',
                    },
                ),
                'resize_mode': (
                    ['fit and pad', 'stretch'],
                    {
                        'default': 'fit and pad',
                        'tooltip': 'fit preserves aspect ratio and pads; stretch fills the cell exactly',
                    },
                ),
                'spatial_padding_color': (
                    ['plot background', 'black', 'white'],
                    {
                        'default': 'plot background',
                        'tooltip': 'color used around a video when its aspect ratio does not fill the common cell resolution',
                    },
                ),
                'fps': (
                    'FLOAT',
                    {
                        'default': 0.0,
                        'min': 0.0,
                        'max': 1000.0,
                        'step': 0.01,
                        'tooltip': 'output FPS override; 0 keeps the FPS from the first cell, or 24 FPS for raw IMAGE batches',
                    },
                ),
            },
            'optional': {
                'plot_config_grid': (
                    'PLOT_CONFIG_GRID',
                    {'tooltip': 'optional: ' + TOOLTIP_PLOT_CONFIG_GRID},
                ),
                'plot_config_header': (
                    'PLOT_CONFIG_HF',
                    {'tooltip': 'optional: plot configuration for the page header'},
                ),
                'plot_config_footer': (
                    'PLOT_CONFIG_HF',
                    {'tooltip': 'optional: plot configuration for the page footer'},
                ),
                'group_dim2_headers': (
                    'BOOLEAN',
                    {
                        'default': False,
                        'tooltip': 'group Cartesian-product DIM2 columns by their first value',
                    },
                ),
                'dim2_group_header_format': (
                    'STRING',
                    {
                        'default': 'Prompt: {dim2_group:.60}…',
                        'tooltip': "group header template. Use '{dim2_group}' for the first Cartesian-product value",
                    },
                ),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('VIDEO', 'IMAGE')
    RETURN_NAMES = ('video', 'plot_frames')
    OUTPUT_TOOLTIPS = (
        'silent or single-audio-track XY plot video',
        'plot frames as an IMAGE batch; connect to Create Video if needed',
    )
    DESCRIPTION = 'Render an XY plot for every frame of a video. Shorter cells can repeat their final frame or use a solid fill; different resolutions are normalized before plotting.'

    def check_lazy_status(self, xy_plot_data, video=None, **kwargs):
        if xy_plot_data.cached_cells is not None:
            return []
        if video is None:
            return ['video']
        return []

    def _load_cached_cells(self, cached_cells, fps):
        cells = []
        for cell in cached_cells:
            plot_data = _plot_data_from_dict(cell['plot_data'])
            record = _XYPlotImageCache(key=cell['cache_key']).load_media(
                fps if fps > 0 else 24.0
            )
            if record is None:
                raise RuntimeError(
                    'A cached XY Plot video cell disappeared; run the plot again to rebuild it'
                )
            cells.append((plot_data, record))
        return cells

    def _select_audio(self, cells, audio_mode, audio_cell, direction):
        if audio_mode == 'silent':
            return None
        if audio_mode == 'first cell':
            selected = cells[0][1].audio
        elif audio_mode == 'first available':
            selected = next(
                (record.audio for _, record in cells if record.audio is not None),
                None,
            )
        else:
            index = _video_cell_indices(cells, audio_cell, direction)
            selected = cells[index][1].audio
            if selected is None:
                raise ValueError(
                    f"Audio cell {audio_cell.upper()} has no audio track. Choose 'first available' or another cell."
                )
        return _copy_audio(selected)

    def _render_frames(
        self,
        cells,
        dim1_header_format,
        dim2_header_format,
        direction,
        temporal_padding,
        resolution_mode,
        resize_mode,
        spatial_padding_color,
        plot_config_grid,
        plot_config_header,
        plot_config_footer,
        group_dim2_headers,
        dim2_group_header_format,
    ):
        first_data = cells[0][0]
        target_frames = max(record.frames.shape[0] for _, record in cells)
        if resolution_mode == 'first cell':
            target_height, target_width = cells[0][1].frames.shape[1:3]
        else:
            target_height = max(record.frames.shape[1] for _, record in cells)
            target_width = max(record.frames.shape[2] for _, record in cells)

        if temporal_padding == 'black':
            temporal_rgb = (0, 0, 0)
        elif temporal_padding == 'white':
            temporal_rgb = (255, 255, 255)
        else:
            temporal_rgb = _background_rgb(
                plot_config_grid.background_color
                if temporal_padding == 'plot background'
                else '#000000'
            )
        if spatial_padding_color == 'black':
            pad_rgb = (0, 0, 0)
        elif spatial_padding_color == 'white':
            pad_rgb = (255, 255, 255)
        else:
            pad_rgb = _background_rgb(plot_config_grid.background_color)
        output_frames = []

        for frame_index in range(target_frames):
            pager = Pager(
                first_data,
                (dim1_header_format, dim2_header_format),
                direction,
                group_dim2_headers,
                dim2_group_header_format,
            )
            for plot_data, record in cells:
                if frame_index < record.frames.shape[0]:
                    image = _video_frame_image(
                        record.frames[frame_index],
                        (target_width, target_height),
                        resize_mode,
                        pad_rgb,
                    )
                elif temporal_padding == 'repeat last frame':
                    image = _video_frame_image(
                        record.frames[-1],
                        (target_width, target_height),
                        resize_mode,
                        pad_rgb,
                    )
                else:
                    image = Image.new(
                        'RGB',
                        (target_width, target_height),
                        color=temporal_rgb,
                    )
                pager.add(plot_data, pillow_to_tensor(image))

            grid = pager.make_grid(
                PlotVars(first_data.current_page + 1, first_data.total_pages),
                plot_config_grid,
                plot_config_header,
                plot_config_footer,
            )
            # Pager.make_grid() already returns a ComfyUI image tensor.
            # Keep the renderer from treating it as a PIL image a second time.
            if grid.shape[-1] == 4:
                grid = grid[..., :3]
            # libx264 requires even frame dimensions. The plot layout can be
            # odd-sized when headers, gaps, or padding produce an odd total.
            pad_height = int(grid.shape[1] % 2)
            pad_width = int(grid.shape[2] % 2)
            if pad_height or pad_width:
                padded = grid.new_empty(
                    (
                        grid.shape[0],
                        grid.shape[1] + pad_height,
                        grid.shape[2] + pad_width,
                        grid.shape[3],
                    )
                )
                pad_rgb = _background_rgb(plot_config_grid.background_color)
                padded[...] = grid.new_tensor(pad_rgb).view(1, 1, 1, 3)
                padded[:, : grid.shape[1], : grid.shape[2], :] = grid
                grid = padded
            output_frames.append(grid)

        return torch.cat(output_frames, dim=0)

    def run(
        self,
        xy_plot_data: XYPlotQueueData,
        video,
        dim1_header_format: str,
        dim2_header_format: str,
        direction: bool,
        audio_mode: str,
        audio_cell: str,
        temporal_padding: str,
        resolution_mode: str,
        resize_mode: str,
        spatial_padding_color: str,
        fps: float,
        plot_config_grid=PlotConfigGridData(),
        plot_config_header=None,
        plot_config_footer=None,
        group_dim2_headers=False,
        dim2_group_header_format='Prompt: {dim2_group:.60}…',
    ):
        if xy_plot_data.cached_cells is not None:
            cells = self._load_cached_cells(xy_plot_data.cached_cells, fps)
        else:
            if xy_plot_data.index == 0 or self.cells is None:
                self.cells = {}
                self.page = xy_plot_data.current_page
            elif self.page != xy_plot_data.current_page:
                self.cells = {}
                self.page = xy_plot_data.current_page
            self.cells[xy_plot_data.index] = (
                xy_plot_data,
                _video_record(video, fps if fps > 0 else 24.0),
            )
            expected = xy_plot_data.dim1.length * xy_plot_data.dim2.length
            if len(self.cells) < expected:
                return {
                    'result': (ExecutionBlocker(None), ExecutionBlocker(None))
                }
            cells = [self.cells[index] for index in range(expected)]

        plot_frames = self._render_frames(
            cells,
            dim1_header_format,
            dim2_header_format,
            direction,
            temporal_padding,
            resolution_mode,
            resize_mode,
            spatial_padding_color,
            plot_config_grid,
            plot_config_header,
            plot_config_footer,
            group_dim2_headers,
            dim2_group_header_format,
        )
        output_fps = fps if fps > 0 else cells[0][1].frame_rate
        audio = self._select_audio(cells, audio_mode, audio_cell, direction)
        output_record = SimpleNamespace(
            frames=plot_frames,
            audio=audio,
            frame_rate=_frame_rate(output_fps),
        )
        return (_video_from_record(output_record), plot_frames)


@register_node('Plot Config: Grid', 'plot')
class PlotConfigGrid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'gap': (
                    'INT',
                    {'default': 20, 'min': 0, 'tooltip': 'gap between grid cells'},
                ),
                'background_color': (
                    'STRING',
                    {
                        'default': '#b9b9b9',
                        'tooltip': 'background color, '
                        + TOOLTIP_COLOR
                        + "\nuse the special value 'transparent' to generate a RGBA image",
                    },
                ),
                'font': (
                    'STRING',
                    {
                        'default': 'Roboto-Regular.ttf',
                        'tooltip': TOOLTIP_FONT,
                    },
                ),
                'font_size': (
                    'INT',
                    {'default': 50, 'min': 1, 'tooltip': TOOLTIP_FONT_SIZE},
                ),
                'font_color': (
                    'STRING',
                    {'default': '#444', 'tooltip': TOOLTIP_FONT_COLOR},
                ),
                'pad_col_headers': (
                    'INT',
                    {
                        'default': 30,
                        'min': 0,
                        'tooltip': 'vertical padding to apply to column headers',
                    },
                ),
                'pad_row_headers': (
                    'INT',
                    {
                        'default': 50,
                        'min': 0,
                        'tooltip': 'horizontal padding to apply to row headers',
                    },
                ),
                'wrap_col_headers': (
                    'INT',
                    {
                        'default': 0,
                        'min': 0,
                        'tooltip': 'if > 0, max number of characters in column headers before wrapping',
                    },
                ),
                'wrap_row_headers': (
                    'INT',
                    {
                        'default': 0,
                        'min': 0,
                        'tooltip': 'if > 0, max number of characters in row headers before wrapping',
                    },
                ),
                'group_gap': (
                    'INT',
                    {
                        'default': 0,
                        'min': 0,
                        'tooltip': 'extra horizontal gap between grouped DIM2 column headers',
                    },
                ),
            },
            'optional': {
                'wrap_col_headers_mode': (
                    ['manual', 'auto'],
                    {
                        'default': 'manual',
                        'tooltip': 'manual uses the character count above; auto measures the selected font against each cell width',
                    },
                ),
                'wrap_row_headers_mode': (
                    ['manual', 'auto'],
                    {
                        'default': 'manual',
                        'tooltip': 'manual uses the character count above; auto measures the selected font against a width proportional to each cell',
                    },
                ),
                'auto_wrap_col_width': (
                    'FLOAT',
                    {
                        'default': 0.9,
                        'min': 0.1,
                        'max': 2.0,
                        'step': 0.05,
                        'tooltip': 'automatic column-header line width as a multiple of the cell width',
                    },
                ),
                'auto_wrap_row_width': (
                    'FLOAT',
                    {
                        'default': 0.9,
                        'min': 0.1,
                        'max': 2.0,
                        'step': 0.05,
                        'tooltip': 'automatic row-header line width as a multiple of the cell width',
                    },
                ),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('PLOT_CONFIG_GRID',)
    RETURN_NAMES = ('plot_config_grid',)
    OUTPUT_TOOLTIPS = (TOOLTIP_PLOT_CONFIG_GRID,)
    DESCRIPTION = 'Various options to customize the grid appearance.'

    def run(
        self,
        gap: int,
        background_color: str,
        font: str,
        font_size: int,
        font_color: str,
        pad_col_headers: int,
        pad_row_headers: int,
        wrap_col_headers: int,
        wrap_row_headers: int,
        group_gap: int,
        wrap_col_headers_mode: str = 'manual',
        wrap_row_headers_mode: str = 'manual',
        auto_wrap_col_width: float = 0.9,
        auto_wrap_row_width: float = 0.9,
    ):
        plot_config_grid = PlotConfigGridData(
            gap=gap,
            group_gap=group_gap,
            background_color=background_color,
            font=font,
            font_size=font_size,
            font_color=font_color,
            pad_col_headers=pad_col_headers,
            pad_row_headers=pad_row_headers,
            wrap_col_headers=wrap_col_headers,
            wrap_row_headers=wrap_row_headers,
            wrap_col_headers_mode=wrap_col_headers_mode,
            wrap_row_headers_mode=wrap_row_headers_mode,
            auto_wrap_col_width=auto_wrap_col_width,
            auto_wrap_row_width=auto_wrap_row_width,
        )
        return (plot_config_grid,)


@register_node('Plot Config: Header/Footer', 'plot')
class PlotConfigHF:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'text_left': (
                    'STRING',
                    {
                        'default': '',
                        'tooltip': 'text to be displayed on the left of the page header or footer\n'
                        + TOOLTIP_HF_PLACEHOLDERS,
                    },
                ),
                'text_center': (
                    'STRING',
                    {
                        'default': '',
                        'tooltip': 'text to be displayed at the center of the page header or footer\n'
                        + TOOLTIP_HF_PLACEHOLDERS,
                    },
                ),
                'text_right': (
                    'STRING',
                    {
                        'default': '',
                        'tooltip': 'text to be displayed on the right of the page header or footer\n'
                        + TOOLTIP_HF_PLACEHOLDERS,
                    },
                ),
                'background_color': (
                    'STRING',
                    {
                        'default': 'transparent',
                        'tooltip': 'background color for the page header or footer, '
                        + TOOLTIP_COLOR
                        + "\nuse the special value 'transparent' to keep the grid background",
                    },
                ),
                'font': (
                    'STRING',
                    {'default': 'Roboto-Bold.ttf', 'tooltip': TOOLTIP_FONT},
                ),
                'font_size': (
                    'INT',
                    {'default': 60, 'min': 1, 'tooltip': TOOLTIP_FONT_SIZE},
                ),
                'font_color': (
                    'STRING',
                    {'default': '#222', 'tooltip': TOOLTIP_FONT_COLOR},
                ),
                'padding': (
                    'INT',
                    {
                        'default': 30,
                        'min': 0,
                        'tooltip': 'padding inside the page header or footer',
                    },
                ),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('PLOT_CONFIG_HF',)
    RETURN_NAMES = ('plot_config_hf',)
    OUTPUT_TOOLTIPS = ('plot configuration for either the page header or footer',)
    DESCRIPTION = (
        'Various options to customize the appearance of the page header or footer.'
    )

    def run(
        self,
        text_left: str,
        text_center: str,
        text_right: str,
        background_color: str,
        font: str,
        font_size: int,
        font_color: str,
        padding: int,
    ):
        plot_config_hf = PlotConfigHFData(
            text_left=text_left,
            text_center=text_center,
            text_right=text_right,
            font=font,
            background_color=background_color,
            font_size=font_size,
            font_color=font_color,
            padding=padding,
        )
        return (plot_config_hf,)


@register_node('XY Plot: Split Data', 'plot')
class XYPlotDataSplit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'xy_plot_data': ('XY_PLOT_DATA', {'tooltip': TOOLTIP_XY_PLOT_DATA}),
            },
        }

    FUNCTION = 'run'
    RETURN_TYPES = ('INT', 'INT', 'BOOLEAN')
    RETURN_NAMES = ('current_page', 'total_pages', 'complete')
    OUTPUT_TOOLTIPS = (
        'current page',
        'total number of pages',
        'is processing complete? (BOOLEAN)',
    )
    DESCRIPTION = 'Split the queue processing data into individual values.\nUseful to customize the filename during saving, for example.'

    def run(self, xy_plot_data: XYPlotQueueData):
        return (
            xy_plot_data.current_page + 1,
            xy_plot_data.total_pages,
            xy_plot_data.complete,
        )
