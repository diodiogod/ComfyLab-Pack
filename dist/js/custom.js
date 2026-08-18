import { slotShapes } from './shared/utils.js';
const customLabels = {
    XYPlotQueue: {
        widget: {
            max_dim1_per_page: 'dim1: max per page',
            max_dim2_per_page: 'dim2: max per page',
        },
        input: {
            dim2: 'dim2 (opt)',
        },
        output: {
            dim1_value: 'dim1 value',
            dim2_value: 'dim2 value',
            xy_plot_data: 'XY plot data',
        },
    },
    XYPlotRender: {
        input: {
            xy_plot_data: 'XY plot data',
            plot_config_grid: 'config: grid (opt)',
            plot_config_header: 'config: header (opt)',
            plot_config_footer: 'config: footer (opt)',
        },
        widget: {
            dim1_header_format: 'dim1: header format',
            dim2_header_format: 'dim2: header format',
        },
    },
    XYPlotVideoCache: {
        input: {
            xy_plot_data: 'XY plot data (opt)',
        },
        widget: {
            fps: 'fallback FPS',
            cache_mode: 'cache mode',
            cache_key: 'cache namespace',
            max_cache_mb: 'max cache (MB)',
        },
    },
    XYPlotVideoRender: {
        input: {
            xy_plot_data: 'XY plot data',
            plot_config_grid: 'config: grid (opt)',
            plot_config_header: 'config: header (opt)',
            plot_config_footer: 'config: footer (opt)',
        },
        widget: {
            dim1_header_format: 'dim1: header format',
            dim2_header_format: 'dim2: header format',
            audio_mode: 'audio',
            audio_cell: 'audio cell',
            temporal_padding: 'short video padding',
            resolution_mode: 'cell resolution',
            resize_mode: 'resize mode',
            spatial_padding_color: 'aspect-ratio padding',
            fps: 'output FPS',
        },
    },
    PlotConfigGrid: {
        widget: {
            gap: 'gap (between cells)',
            background_color: 'background color',
            font_size: 'font size',
            font_color: 'font color',
            pad_col_headers: 'col headers: padding',
            pad_row_headers: 'row headers: padding',
            wrap_col_headers: 'col headers: wrap',
            wrap_row_headers: 'row headers: wrap',
        },
        output: {
            plot_config_grid: 'config: grid',
        },
    },
    PlotConfigHF: {
        widget: {
            text_left: 'text (left)',
            text_center: 'text (center)',
            text_right: 'text (right)',
            background_color: 'background color',
            font: 'font',
            font_size: 'font size',
            font_color: 'font color',
            padding: 'padding',
        },
        output: {
            plot_config_hf: 'config: header / footer',
        },
    },
    XYPlotDataSplit: {
        input: {
            xy_plot_data: 'XY plot data',
        },
        output: {
            current_page: 'current page',
            total_pages: 'total pages',
            complete: 'complete?',
        },
    },
    ListFromMultiline: {
        widget: {
            strip_comments: 'strip comments',
        },
    },
    ListFromFile: {
        widget: {
            file_path: 'path',
            strip_comments: 'strip comments',
        },
    },
    ListLimit: {
        widget: {
            nb_elements: 'nb of elements',
        },
    },
    ListCheckpoints: {
        widget: {
            with_extension: 'with file extension?',
        },
    },
    ListLoras: {
        widget: {
            with_extension: 'with file extension?',
        },
    },
    OutputConfigBackend: {
        widget: {
            file_path: 'JSON or YAML file (backend)',
            show_config: 'show config',
            error_display: 'display errors',
        },
    },
    OutputConfigLocal: {
        widget: {
            error_display: 'display errors',
            show_config: 'show config',
        },
    },
    GenericQueue: {
        input: {
            input_list: 'input list',
        },
    },
    FileQueue: {
        widget: {
            pattern: 'pattern(s)',
            recursive: 'recursive?',
            with_extension: 'with extension?',
        },
        output: {
            full_path: 'full path',
            relative_path: 'relative path',
        },
    },
    ImageQueue: {
        widget: {
            pattern: 'pattern(s)',
            recursive: 'recursive?',
            with_extension: 'with extension?',
        },
        output: {
            full_path: 'full path',
            relative_path: 'relative path',
        },
    },
    InputFolder: {
        widget: {
            path: 'folder path',
            check_exists: 'check folder existence',
        },
    },
    InputMultiline: {
        widget: {
            strip_comments: 'strip comments',
        },
    },
    LoadImageRGBA: {
        widget: {
            with_extension: 'with extension?',
            mask_precision: 'mask method',
        },
    },
    ConvertToAny: {
        output: {
            value_any: 'value (Any)',
        },
    },
    ResolutionToDims: {
        widget: {
            scale_factor: 'scale factor',
        },
    },
};
const customLooks = {
    XYPlotQueue: {
        input: {
            dim1: {
                shape: 'grid',
            },
            dim2: {
                shape: 'grid',
            },
        },
        output: {
            xy_plot_data: {
                color_off: '#52395B',
                color_on: '#9521c3',
            },
        },
    },
    XYPlotRender: {
        input: {
            xy_plot_data: {
                color_off: '#52395B',
                color_on: '#9521c3',
            },
        },
    },
    XYPlotDataSplit: {
        input: {
            xy_plot_data: {
                color_off: '#52395B',
                color_on: '#9521c3',
            },
        },
    },
    ListFromString: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListFromMultiline: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListFromFile: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListFromElements: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListMerge: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListLimit: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListCheckpoints: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListLoras: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListSamplers: {
        output: {
            list: { shape: 'grid' },
        },
    },
    ListSchedulers: {
        output: {
            list: { shape: 'grid' },
        },
    },
};
function setNodeLabels(nodeType, nodeData) {
    const nodeName = nodeData.name;
    if (!nodeName || !Object.keys(customLabels).includes(nodeName))
        return;
    const original_onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
        const r = original_onNodeCreated.apply(this, args);
        for (const [scope, scope_entry] of Object.entries(customLabels[nodeName])) {
            for (const [name, label] of Object.entries(scope_entry)) {
                let nodeItems = undefined;
                switch (scope) {
                    case 'input':
                        nodeItems = this.inputs;
                        break;
                    case 'output':
                        nodeItems = this.outputs;
                        break;
                    case 'widget':
                        nodeItems = this.widgets;
                        break;
                }
                const item = nodeItems?.find((i) => i.name === name);
                if (item)
                    item.label = label;
            }
        }
        return r;
    };
}
function setNodeLooks(nodeType, nodeData) {
    const nodeName = nodeData.name;
    if (!nodeName || !Object.keys(customLooks).includes(nodeName))
        return;
    const original_onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
        const r = original_onNodeCreated.apply(this, args);
        for (const [scope, scope_entry] of Object.entries(customLooks[nodeName])) {
            for (const [name, lookEntry] of Object.entries(scope_entry)) {
                let nodeItems = undefined;
                switch (scope) {
                    case 'input':
                        nodeItems = this.inputs;
                        break;
                    case 'output':
                        nodeItems = this.outputs;
                        break;
                }
                const item = nodeItems?.find((i) => i.name === name);
                if (item) {
                    for (const [lookName, lookValue] of Object.entries(lookEntry)) {
                        if (lookName === 'shape')
                            item[lookName] = slotShapes[lookValue];
                        else
                            item[lookName] = lookValue;
                    }
                }
            }
        }
        return r;
    };
}
export function customizeNode(nodeType, nodeData) {
    setNodeLabels(nodeType, nodeData);
    setNodeLooks(nodeType, nodeData);
}
