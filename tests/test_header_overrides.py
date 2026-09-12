import unittest

import torch

from src.python.shared.pager import Pager
from src.python.shared.plot_data import (
    DimData,
    PlotConfigGridData,
    PlotHeaderOverrideRule,
    PlotHeaderOverridesData,
    PlotHeaderText,
    PlotVars,
    XYPlotQueueData,
)


class HeaderOverrideTests(unittest.TestCase):
    def _data(self, dim1='epoch_1.safetensors', dim2='prompt'):
        return XYPlotQueueData(0, 0, 1, True, DimData(0, 1, dim1), DimData(0, 1, dim2))

    def test_append_matches_raw_value_and_keeps_base_text(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule('dim1', 'contains', 'epoch_1', 'append', 'OLD', '#ff0000')
        ])
        pager = Pager(self._data(), ('{dim1.removesuffix(".safetensors")}', '{dim2}'), header_overrides=overrides)
        pager.add(self._data(), torch.zeros((1, 16, 16, 3)))
        header = pager.dim1.headers[0]
        self.assertIsInstance(header, PlotHeaderText)
        self.assertEqual(header.plain_text, 'epoch_1 — OLD')
        self.assertEqual(header.segments[-1].color, '#ff0000')

    def test_non_matching_header_stays_plain(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule('dim1', 'exact', 'epoch_2', 'append', 'OLD', 'red')
        ])
        pager = Pager(self._data(), ('{dim1}', '{dim2}'), header_overrides=overrides)
        pager.add(self._data(), torch.zeros((1, 16, 16, 3)))
        self.assertEqual(pager.dim1.headers[0], 'epoch_1.safetensors')

    def test_styled_header_renders_with_automatic_wrap(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule('dim2', 'exact', 'prompt', 'append', 'OLD', '#ff0000')
        ])
        data = self._data()
        pager = Pager(data, ('{dim1}', 'A deliberately long {dim2} header'), header_overrides=overrides)
        pager.add(data, torch.zeros((1, 64, 64, 3)))
        result = pager.make_grid(
            PlotVars(1, 1),
            PlotConfigGridData(font_size=12, wrap_col_headers_mode='auto'),
        )
        self.assertEqual(result.ndim, 4)
        self.assertGreater(result.shape[1], 64)

    def test_regex_highlight_preserves_and_colors_exact_match(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule(
                'dim1',
                'regex',
                r'-v\.?\d+(?:\.\d+)?',
                'highlight match',
                '',
                '#ff0000',
            )
        ])
        data = self._data(dim1='Sweaty_Shirt-v.1.0.safetensors')
        pager = Pager(
            data,
            ('{dim1.removesuffix(".safetensors")}', '{dim2}'),
            header_overrides=overrides,
        )
        pager.add(data, torch.zeros((1, 16, 16, 3)))
        header = pager.dim1.headers[0]
        self.assertEqual(header.plain_text, 'Sweaty_Shirt-v.1.0')
        self.assertEqual(
            [(segment.text, segment.color) for segment in header.segments],
            [('Sweaty_Shirt', None), ('-v.1.0', '#ff0000')],
        )

    def test_regex_highlight_searches_formatted_header_not_raw_value(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule(
                'dim1',
                'regex',
                r'-v\.?\d+(?:\.\d+)?',
                'highlight match',
                '',
                '#ff0000',
            )
        ])
        data = self._data(dim1='raw value without a version')
        pager = Pager(
            data,
            ('Male_Anatomy-v2.02', '{dim2}'),
            header_overrides=overrides,
        )
        pager.add(data, torch.zeros((1, 16, 16, 3)))
        header = pager.dim1.headers[0]
        self.assertEqual(
            [(segment.text, segment.color) for segment in header.segments],
            [('Male_Anatomy', None), ('-v2.02', '#ff0000')],
        )

    def test_invalid_regex_has_friendly_error(self):
        overrides = PlotHeaderOverridesData([
            PlotHeaderOverrideRule(
                'dim1', 'regex', '[', 'highlight match', '', 'red'
            )
        ])
        data = self._data()
        pager = Pager(data, ('{dim1}', '{dim2}'), header_overrides=overrides)
        with self.assertRaisesRegex(ValueError, 'Invalid Header Override regex'):
            pager.add(data, torch.zeros((1, 16, 16, 3)))


if __name__ == '__main__':
    unittest.main()
