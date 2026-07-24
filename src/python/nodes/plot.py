from comfy_execution.graph import ExecutionBlocker  # type: ignore
import folder_paths  # type: ignore
import torch  # type: ignore
from PIL import Image
from math import ceil
import hashlib
import json
import os
import shutil
import time

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
from ..shared.utils import pillow_to_tensor

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


def _canonical_prompt_node(prompt, node_id, seen=None, queue_index=None):
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
            return [
                'link',
                _canonical_prompt_node(prompt, value[0], seen, queue_index),
                value[1],
            ]
        if isinstance(value, dict):
            return {key: canonical_input(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [canonical_input(item) for item in value]
        return value

    inputs = dict(node.get('inputs', {}))
    index = inputs.get('index', 0)
    if (
        node.get('class_type') == 'XYPlotQueue'
        and queue_index is not None
    ):
        inputs['index'] = queue_index
    elif (
        node.get('class_type') == 'XYPlotQueue'
        and isinstance(index, (int, float))
        and index < 0
    ):
        inputs['index'] = 0
    return [
        node.get('class_type'),
        {key: canonical_input(inputs[key]) for key in sorted(inputs)},
    ]


class _XYPlotImageCache:
    def __init__(self, prompt=None, unique_id=None, cache_key='', queue_index=None, key=None):
        self.root = os.path.join(
            folder_paths.get_user_directory(), 'comfylab', 'xy_cache'
        )
        self.legacy_root = os.path.join(
            folder_paths.get_temp_directory(), 'comfylab_xy_cache'
        )
        if key is not None:
            self.key = key
            self._migrate_legacy_entry()
            return
        node = prompt.get(str(unique_id), {})
        image_link = node.get('inputs', {}).get('image')
        if not (
            isinstance(image_link, list)
            and len(image_link) == 2
            and str(image_link[0]) in prompt
        ):
            self.key = None
            return
        signature = [
            _canonical_prompt_node(prompt, image_link[0], queue_index=queue_index),
            image_link[1],
            cache_key,
        ]
        encoded = json.dumps(
            signature, sort_keys=True, separators=(',', ':'), ensure_ascii=False
        ).encode('utf-8')
        self.key = hashlib.sha256(encoded).hexdigest()
        self._migrate_legacy_entry()

    def _migrate_legacy_entry(self):
        if (
            self.key is None
            or os.path.isfile(self.manifest_path)
            or not os.path.isdir(self.legacy_root)
        ):
            return
        names = [
            name
            for name in os.listdir(self.legacy_root)
            if name == self.key + '.json' or name.startswith(self.key + '.')
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

    def load(self):
        manifest = self.load_manifest()
        if manifest is None:
            return None
        try:
            frames = []
            for index in range(manifest['frames']):
                path = os.path.join(self.root, f'{self.key}.{index}.png')
                with Image.open(path) as image:
                    frames.append(pillow_to_tensor(image.convert(manifest['mode'])))
            now = time.time()
            os.utime(self.manifest_path, (now, now))
            return torch.cat(frames, dim=0)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            self.remove()
            return None

    def load_manifest(self):
        if self.key is None or not os.path.isfile(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (OSError, ValueError, json.JSONDecodeError):
            self.remove()
            return None

    def save(self, image, max_cache_mb, xy_plot_data=None):
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
        if self.key is None or not os.path.isdir(self.root):
            return
        for name in os.listdir(self.root):
            if name == self.key + '.json' or name.startswith(self.key + '.'):
                try:
                    os.remove(os.path.join(self.root, name))
                except FileNotFoundError:
                    pass

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
        if node.get('class_type') != 'XYPlotImageCache':
            continue
        inputs = node.get('inputs', {})
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
            cache = _XYPlotImageCache(
                prompt,
                node_id,
                inputs.get('cache_key', ''),
                queue_index=index,
            )
            manifest = cache.load_manifest()
            plot_data = manifest.get('xy_plot_data') if manifest else None
            if manifest is not None and (
                manifest.get('frames', 0) < 1
                or not all(
                    os.path.isfile(
                        os.path.join(cache.root, f"{cache.key}.{frame}.png")
                    )
                    for frame in range(manifest['frames'])
                )
            ):
                manifest = None
                plot_data = None
            if manifest is not None and plot_data is None:
                dim1_index = int(index / len(dim2))
                dim2_index = index % len(dim2)
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
            if (
                plot_data is None
                or plot_data.get('total_pages') != 1
                or plot_data.get('index') != index
            ):
                cells = []
                break
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
                        'tooltip': 'use cache skips the image branch when the same generation inputs were cached',
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
    DESCRIPTION = 'Cache each generated plot image in the ComfyUI user directory. Place this node immediately before XY Plot: Render to re-render plot styling without sampling again, including after restarting ComfyUI.'

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
        cache = _XYPlotImageCache(prompt, unique_id, cache_key)
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

        cache = _XYPlotImageCache(prompt, unique_id, cache_key)
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
                        'tooltip': "template text to be displayed as dim1 header.\nthe '{dim1'} placeholder will be replaced by the current value.\nUse '\\n' for multiline text.",
                    },
                ),
                'dim2_header_format': (
                    'STRING',
                    {
                        'default': '{dim2}',
                        'tooltip': "template text to be displayed as dim2 header.\nthe '{dim2'} placeholder will be replaced by the current value.\nUse '\\n' for multiline text.",
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
