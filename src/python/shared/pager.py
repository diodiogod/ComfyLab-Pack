from types import SimpleNamespace
import re
import torch  # type: ignore
from .utils import tensor_to_pillow, pillow_to_tensor
from .plot_data import (
    XYPlotQueueData,
    PlotConfigGridData,
    PlotConfigHFData,
    PlotVars,
    PlotHeaderOverridesData,
    PlotHeaderSegment,
    PlotHeaderText,
)
from .grid import Grid
from .formatting import format_string


class Pager:
    def __init__(
        self,
        xy_plot_data: XYPlotQueueData,
        header_formats: tuple[str, str],
        dim1_as_rows: bool = True,
        group_dim2_headers: bool = False,
        dim2_group_header_format: str = '{dim2_group}',
        header_overrides: PlotHeaderOverridesData = None,
    ):
        self.dim1_as_rows = dim1_as_rows
        self.header_formats = header_formats
        self.group_dim2_headers = group_dim2_headers
        self.dim2_group_header_format = dim2_group_header_format
        self.header_overrides = header_overrides
        self.dim1 = SimpleNamespace(
            **{'length': xy_plot_data.dim1.length, 'headers': []}
        )
        self.dim2 = SimpleNamespace(
            **{
                'length': xy_plot_data.dim2.length,
                'headers': [],
                'values': [None] * xy_plot_data.dim2.length,
            }
        )
        # initialize the 2D list
        if dim1_as_rows:
            self.image_matrix = [
                [None] * self.dim2.length for i in range(self.dim1.length)
            ]
        else:
            self.image_matrix = [
                [None] * self.dim1.length for i in range(self.dim2.length)
            ]
        self.accumulated = 0

    @property
    def expected(self) -> int:
        return self.dim1.length * self.dim2.length

    @property
    def complete(self) -> bool:
        return self.accumulated >= self.expected

    # calculate the coordinates in dim1 and dim2, given a sequential index
    def get_coords(self, index: int) -> tuple[int, int]:
        if self.dim1_as_rows:
            return (int(index / self.dim2.length), int(index % self.dim2.length))
        else:
            return (int(index % self.dim2.length), int(index / self.dim2.length))

    def add(self, xy_plot_data: XYPlotQueueData, tensor: torch.Tensor):
        # store dim1 / dim2 headers if not known yet
        # TODO: catch exception and set default header?
        if xy_plot_data.dim1.index >= len(self.dim1.headers):
            header = format_string(self.header_formats[0], dim1=xy_plot_data.dim1.value)
            self.dim1.headers.append(self._apply_overrides('dim1', xy_plot_data.dim1.value, header))
        if xy_plot_data.dim2.index >= len(self.dim2.headers):
            header = format_string(self.header_formats[1], dim2=xy_plot_data.dim2.value)
            self.dim2.headers.append(self._apply_overrides('dim2', xy_plot_data.dim2.value, header))
            self.dim2.values[xy_plot_data.dim2.index] = xy_plot_data.dim2.value

        # store the tensor as image
        x, y = self.get_coords(xy_plot_data.index)
        self.image_matrix[x][y] = tensor_to_pillow(tensor)
        self.accumulated += 1

        # return complete status
        return self.complete

    def make_grid(
        self,
        plot_vars: PlotVars,
        plot_config_grid: PlotConfigGridData = PlotConfigGridData(),
        plot_config_header: PlotConfigHFData = None,
        plot_config_footer: PlotConfigHFData = None,
    ) -> torch.Tensor:
        grid = Grid(plot_config_grid, plot_config_header, plot_config_footer)
        if self.dim1_as_rows:
            col_group_headers = self._dim2_group_headers()
            grid_image = grid.make(
                self.image_matrix,
                self.dim2.headers,
                self.dim1.headers,
                plot_vars,
                col_group_headers,
            )
        else:
            grid_image = grid.make(
                self.image_matrix, self.dim1.headers, self.dim2.headers, plot_vars
            )
        return pillow_to_tensor(grid_image)

    def _dim2_group_headers(self):
        if not self.group_dim2_headers:
            return None
        if not all(
            isinstance(value, (list, tuple)) and len(value) == 2
            for value in self.dim2.values
        ):
            return None

        groups = []
        for index, value in enumerate(self.dim2.values):
            group_value = value[0]
            if groups and groups[-1][0] == group_value:
                groups[-1][2] += 1
            else:
                label = format_string(self.dim2_group_header_format, dim2_group=group_value)
                label = self._apply_overrides('dim2 group', group_value, label)
                groups.append([group_value, index, 1, label])
        return [(group[3], group[1], group[2]) for group in groups]

    def _apply_overrides(self, dimension, raw_value, header):
        if not self.header_overrides:
            return header
        result = PlotHeaderText([PlotHeaderSegment(str(header))])
        raw_text = str(raw_value)
        for rule in self.header_overrides.rules:
            if rule.dimension != dimension:
                continue
            regex = None
            if rule.match_mode == 'regex':
                try:
                    regex = re.compile(
                        rule.match_value,
                        0 if rule.case_sensitive else re.IGNORECASE,
                    )
                except re.error as exc:
                    raise ValueError(
                        f"Invalid Header Override regex '{rule.match_value}': {exc}"
                    ) from exc
                matches = regex.search(raw_text) is not None
            else:
                candidate = raw_text if rule.case_sensitive else raw_text.casefold()
                expected = rule.match_value if rule.case_sensitive else rule.match_value.casefold()
                matches = (
                    candidate == expected
                    if rule.match_mode == 'exact'
                    else expected in candidate
                )
            if rule.action == 'highlight match':
                # Highlighting is visual, so search the final formatted header.
                # The raw DIM value may be a tuple or may have been transformed
                # by the header format before it reaches the plot.
                result = self._highlight_rule_matches(result, rule, regex)
                continue
            if not matches:
                continue
            text = format_string(rule.text, value=raw_value)
            styled = PlotHeaderSegment(text, rule.color)
            if rule.action == 'replace':
                result.segments = [styled]
            elif rule.action == 'prepend':
                result.segments = [styled, PlotHeaderSegment(rule.separator)] + result.segments
            else:
                result.segments += [PlotHeaderSegment(rule.separator), styled]
        return result if any(segment.color for segment in result.segments) else header

    def _highlight_rule_matches(self, header, rule, regex=None):
        text = header.plain_text
        if rule.match_mode == 'regex':
            ranges = [
                match.span()
                for match in regex.finditer(text)
                if match.start() != match.end()
            ]
        else:
            needle = rule.match_value
            if not needle:
                return header
            searchable = text if rule.case_sensitive else text.casefold()
            expected = needle if rule.case_sensitive else needle.casefold()
            if rule.match_mode == 'exact':
                if searchable != expected:
                    return header
                ranges = [(0, len(text))]
            else:
                ranges = []
                start = 0
                while True:
                    index = searchable.find(expected, start)
                    if index < 0:
                        break
                    ranges.append((index, index + len(needle)))
                    start = index + len(needle)

        if not ranges:
            return header

        colors = []
        for segment in header.segments:
            colors.extend([segment.color] * len(segment.text))
        for start, end in ranges:
            colors[start:end] = [rule.color] * (end - start)

        segments = []
        for char, color in zip(text, colors):
            if segments and segments[-1].color == color:
                segments[-1].text += char
            else:
                segments.append(PlotHeaderSegment(char, color))
        return PlotHeaderText(segments)
