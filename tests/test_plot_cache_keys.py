import ast
from pathlib import Path


def _load_canonical_prompt_node():
    path = Path(__file__).parents[1] / 'src/python/nodes/plot.py'
    module = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {
        '_canonical_runtime_value',
        '_canonical_prompt_node',
    }
    namespace = {}
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    exec(compile(ast.Module(functions, type_ignores=[]), path, 'exec'), namespace)
    return namespace['_canonical_prompt_node']


def _prompt(displayed_text, live_text='lora-a.safetensors'):
    return {
        'queue': {
            'class_type': 'XYPlotQueue',
            'inputs': {'dim1': ['prompts', 0], 'dim2': ['loras', 0], 'index': 0},
        },
        'extract': {
            'class_type': 'RegexExtract',
            'inputs': {'text': ['queue', 2]},
        },
        'show': {
            'class_type': 'ShowText|pysssss',
            'inputs': {'text': ['extract', 0], 'text_0': displayed_text},
        },
        'loader': {
            'class_type': 'LoraLoaderModelOnly',
            'inputs': {'lora_name': ['show', 0], 'model': ['model', 0]},
        },
        'model': {'class_type': 'ModelLoader', 'inputs': {'name': 'model.safetensors'}},
    }


def test_show_text_display_history_does_not_change_cell_signature():
    canonical = _load_canonical_prompt_node()
    first = canonical(_prompt('previous-lora.safetensors'), 'loader', cell_values=('p', 'lora-a'))
    second = canonical(_prompt('another-lora.safetensors'), 'loader', cell_values=('p', 'lora-a'))

    assert first == second


def test_actual_xy_cell_value_still_changes_signature_through_show_text():
    canonical = _load_canonical_prompt_node()
    first = canonical(_prompt('previous-lora.safetensors'), 'loader', cell_values=('p', 'lora-a'))
    second = canonical(_prompt('previous-lora.safetensors'), 'loader', cell_values=('p', 'lora-b'))

    assert first != second


def test_unconnected_show_text_widget_remains_part_of_signature():
    canonical = _load_canonical_prompt_node()
    first_prompt = _prompt('first value')
    second_prompt = _prompt('second value')
    del first_prompt['show']['inputs']['text']
    del second_prompt['show']['inputs']['text']

    first = canonical(first_prompt, 'show')
    second = canonical(second_prompt, 'show')

    assert first != second


def test_core_preview_any_has_no_saved_display_state_to_fingerprint():
    canonical = _load_canonical_prompt_node()
    prompt = {
        'queue': {
            'class_type': 'XYPlotQueue',
            'inputs': {'dim1': ['prompts', 0], 'dim2': ['loras', 0], 'index': 0},
        },
        'preview': {
            'class_type': 'PreviewAny',
            'inputs': {'source': ['queue', 2]},
        },
    }

    first = canonical(prompt, 'preview', cell_values=('p', 'lora-a'))
    second = canonical(prompt, 'preview', cell_values=('p', 'lora-b'))

    assert first != second
