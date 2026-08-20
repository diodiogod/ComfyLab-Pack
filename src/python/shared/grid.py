from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap
import math
from dataclasses import asdict

from .plot_data import PlotConfigGridData, PlotConfigHFData, PlotVars, PlotHeaderText, PlotHeaderSegment
from .formatting import format_string

STATIC_DIR = (Path(__file__).parent.parent.parent.parent / 'static').resolve()


class Grid:
    def __init__(
        self,
        plot_config_grid: PlotConfigGridData,
        plot_config_header: PlotConfigHFData = None,
        plot_config_footer: PlotConfigHFData = None,
    ):
        self.config_grid = plot_config_grid
        self.config_header = plot_config_header
        self.config_footer = plot_config_footer
        self.headers = ([], [])
        self.dims = (0, 0)

    def make(
        self,
        image_matrix: list[list[Image.Image]],
        col_headers: list[str],
        row_headers: list[str],
        plot_vars: PlotVars,
        col_group_headers=None,
    ) -> Image.Image:
        # keep a track to reuse later
        self.headers = (col_headers, row_headers)
        self.dims = (len(col_headers), len(row_headers))
        cols, rows = self.dims
        self.max_cell_size = self._calc_max_cell_size(image_matrix)
        self.col_group_starts = (
            [start for _, start, _ in col_group_headers[1:]]
            if col_group_headers
            else []
        )

        # build the grid image
        grid_size = (
            self.max_cell_size[0] * cols
            + (cols - 1) * self.config_grid.gap
            + len(self.col_group_starts) * self.config_grid.group_gap,
            self.max_cell_size[1] * rows + (rows - 1) * self.config_grid.gap,
        )
        grid_image = self._create_image(grid_size, self.config_grid.background_color)

        for r, row in enumerate(image_matrix):
            for c, image in enumerate(row):
                x = self._column_x(c) + int(
                    (self.max_cell_size[0] - image.size[0]) / 2
                )
                y = r * (self.max_cell_size[1] + self.config_grid.gap) + int(
                    (self.max_cell_size[1] - image.size[1]) / 2
                )
                grid_image.paste(image, (x, y))

        # add headers; if header format for dim1/dim2 is empty, they will be silently ignored
        grid_image = self._add_headers(grid_image, col_group_headers)

        # add page header / footer
        grid_image = self._add_page_hf(grid_image, plot_vars)

        return grid_image

    def _column_x(self, col: int) -> int:
        group_offset = (
            sum(start <= col for start in self.col_group_starts)
            * self.config_grid.group_gap
        )
        return col * (self.max_cell_size[0] + self.config_grid.gap) + group_offset

    def _calc_max_cell_size(
        self, image_matrix: list[list[Image.Image]]
    ) -> tuple[int, int]:
        # scan all images to get max width / max height
        max_size = (-1, -1)
        # TODO: allow different row heights / col widths?
        for row in image_matrix:
            for image in row:
                w, h = image.size
                max_size = (max(max_size[0], w), max(max_size[1], h))

        return max_size

    def _create_image(self, size: tuple[int, int], bg_color: str) -> Image.Image:
        if bg_color == 'transparent':
            image = Image.new('RGBA', (size[0], size[1]))
        else:
            image = Image.new(
                'RGB',
                (size[0], size[1]),
                color=bg_color,
            )
        return image

    def _add_headers(self, grid_image: Image.Image, col_group_headers=None):
        # Load the selected font before wrapping so automatic mode can measure
        # the real rendered text instead of estimating from character counts.
        font = self._load_font(self.config_grid.font, self.config_grid.font_size)

        # normalize the headers, to respect the wrap configs
        self.headers = (
            self._normalize_headers(
                self.headers[0],
                self.config_grid.wrap_col_headers,
                self.config_grid.wrap_col_headers_mode,
                font,
                self.max_cell_size[0] * self.config_grid.auto_wrap_col_width,
            ),
            self._normalize_headers(
                self.headers[1],
                self.config_grid.wrap_row_headers,
                self.config_grid.wrap_row_headers_mode,
                font,
                self.max_cell_size[0] * self.config_grid.auto_wrap_row_width,
            ),
        )
        if col_group_headers:
            col_group_headers = [
                (
                    self._normalize_headers(
                        [header],
                        self.config_grid.wrap_col_headers,
                        self.config_grid.wrap_col_headers_mode,
                        font,
                        (
                            span * self.max_cell_size[0]
                            + (span - 1) * self.config_grid.gap
                        )
                        * self.config_grid.auto_wrap_col_width,
                    )[0],
                    start,
                    span,
                )
                for header, start, span in col_group_headers
            ]

        # calculate height and width of headers at top and left
        top_margin, left_margin = self._calc_headers_margins(
            font, col_group_headers
        )
        group_margin = self._calc_header_tier_height(
            font,
            [header for header, _, _ in col_group_headers]
            if col_group_headers
            else [],
        )

        # build the new image and paste the existing grid
        image = self._create_image(
            (grid_image.size[0] + left_margin, grid_image.size[1] + top_margin),
            self.config_grid.background_color,
        )
        draw = ImageDraw.Draw(image)
        draw.font = font
        image.paste(
            grid_image,
            (image.size[0] - grid_image.size[0], image.size[1] - grid_image.size[1]),
        )

        # add the headers
        if col_group_headers:
            for header, start, span in col_group_headers:
                group_width = (
                    span * self.max_cell_size[0]
                    + (span - 1) * self.config_grid.gap
                )
                pos_x = (
                    left_margin
                    + self._column_x(start)
                    + group_width / 2
                )
                self._draw_header(
                    draw,
                    (pos_x, group_margin / 2),
                    header,
                    font,
                    self.config_grid.font_color,
                )
        for col, header in enumerate(self.headers[0]):
            pos_x = (
                left_margin
                + self._column_x(col)
                + self.max_cell_size[0] / 2
            )
            pos_y = group_margin + (top_margin - group_margin) / 2
            self._draw_header(
                draw, (pos_x, pos_y), header, font, self.config_grid.font_color
            )
        for row, header in enumerate(self.headers[1]):
            pos_x = left_margin / 2
            pos_y = (
                top_margin
                + row * (self.config_grid.gap + self.max_cell_size[1])
                + self.max_cell_size[1] / 2
            )
            self._draw_header(
                draw, (pos_x, pos_y), header, font, self.config_grid.font_color
            )
        return image

    def _load_font(self, font_name: str, font_size: int) -> ImageFont:
        full_path = (
            STATIC_DIR / font_name
            if not Path(font_name).is_absolute()
            else Path(font_name)
        )
        try:
            full_path = full_path.resolve(strict=True)
        except:
            raise Exception("TTF font not found: '{}'".format(full_path))
        font = ImageFont.truetype(str(full_path), size=font_size)
        return font

    def _normalize_headers(
        self,
        headers: list[str],
        wrap: int,
        wrap_mode: str = 'manual',
        font: ImageFont = None,
        max_width: float = 0,
    ) -> list[str]:
        normalized = []
        for header in headers:
            if isinstance(header, PlotHeaderText):
                header = PlotHeaderText([
                    PlotHeaderSegment(segment.text.replace(r'\n', '\n'), segment.color)
                    for segment in header.segments
                ])
                plain = header.plain_text
                if wrap_mode == 'auto' and font and max_width > 0:
                    wrapped = self._wrap_text_to_width(plain, font, max_width)
                    header = self._restore_styled_wrap(header, wrapped)
                elif wrap > 0:
                    wrapped = textwrap.fill(plain, wrap, break_on_hyphens=True)
                    header = self._restore_styled_wrap(header, wrapped)
                normalized.append(header)
                continue
            header = header.replace(r'\n', '\n')
            if wrap_mode == 'auto' and font and max_width > 0:
                header = self._wrap_text_to_width(header, font, max_width)
            elif wrap > 0:
                header = textwrap.fill(header, wrap, break_on_hyphens=True)
            normalized.append(header)
        return normalized

    def _plain(self, header):
        return header.plain_text if isinstance(header, PlotHeaderText) else header

    def _restore_styled_wrap(self, header, wrapped):
        """Copy colors from the original visible characters onto wrapped text."""
        colored = []
        for segment in header.segments:
            colored.extend((char, segment.color) for char in segment.text)
        result = []
        source = 0
        for char in wrapped:
            if char == '\n':
                result.append(PlotHeaderSegment(char, None))
                while source < len(colored) and colored[source][0].isspace():
                    source += 1
                continue
            while source < len(colored) and colored[source][0] != char:
                source += 1
            color = colored[source][1] if source < len(colored) else None
            source += 1
            if result and result[-1].color == color:
                result[-1].text += char
            else:
                result.append(PlotHeaderSegment(char, color))
        return PlotHeaderText(result)

    def _wrap_text_to_width(
        self, text: str, font: ImageFont, max_width: float
    ) -> str:
        """Wrap text to a rendered pixel width while preserving explicit lines."""
        return '\n'.join(
            self._wrap_line_to_width(line, font, max_width)
            for line in text.split('\n')
        )

    def _wrap_line_to_width(
        self, line: str, font: ImageFont, max_width: float
    ) -> str:
        if not line or font.getlength(line) <= max_width:
            return line

        output = []
        current = ''
        for word in line.split():
            candidate = word if not current else f'{current} {word}'
            if font.getlength(candidate) <= max_width:
                current = candidate
                continue

            if current:
                output.append(current)
                current = ''

            # Split exceptionally long words so automatic wrapping always fits.
            chunk = ''
            for char in word:
                candidate = chunk + char
                if chunk and font.getlength(candidate) > max_width:
                    output.append(chunk)
                    chunk = char
                else:
                    chunk = candidate
            current = chunk

        if current:
            output.append(current)
        return '\n'.join(output)

    def _calc_header_tier_height(self, font: ImageFont, headers):
        height = 0
        for header in headers:
            if header:
                height = max(height, math.ceil(len(self._plain(header).split('\n')) * font.size))
        if height > 0:
            height += 2 * self.config_grid.pad_col_headers
        return height

    def _calc_headers_margins(self, font: ImageFont, col_group_headers=None):
        top_margin = self._calc_header_tier_height(font, self.headers[0])
        if col_group_headers:
            top_margin += self._calc_header_tier_height(
                font, [header for header, _, _ in col_group_headers]
            )

        left_margin = 0
        for header in self.headers[1]:
            header = self._plain(header)
            if header == '':
                continue
            for line in header.split('\n'):
                left_margin = max(left_margin, math.ceil(font.getlength(line)))
        # add padding if there is something to display
        if left_margin > 0:
            left_margin = left_margin + 2 * self.config_grid.pad_row_headers

        return (top_margin, left_margin)

    def _draw_header(
        self,
        draw: ImageDraw,
        pos: tuple[int, int],
        text: str,
        font: ImageFont,
        font_color,
    ):
        if isinstance(text, PlotHeaderText):
            self._draw_styled_header(draw, pos, text, font, font_color)
            return
        # see: https://github.com/python-pillow/Pillow/discussions/7914#discussioncomment-8950499
        draw.multiline_text(
            pos, text, font=font, fill=font_color, align='center', anchor='mm'
        )

    def _draw_styled_header(self, draw, pos, text, font, default_color):
        lines = [[]]
        for segment in text.segments:
            parts = segment.text.split('\n')
            for index, part in enumerate(parts):
                if part:
                    lines[-1].append((part, segment.color or default_color))
                if index < len(parts) - 1:
                    lines.append([])
        start_y = pos[1] - (len(lines) - 1) * font.size / 2
        for line_index, line in enumerate(lines):
            width = sum(font.getlength(part) for part, _ in line)
            x = pos[0] - width / 2
            y = start_y + line_index * font.size
            for part, color in line:
                draw.text((x, y), part, font=font, fill=color, anchor='lm')
                x += font.getlength(part)

    def _add_page_hf(self, grid_image: Image.Image, plot_vars: PlotVars) -> Image.Image:
        grid_width, grid_height = grid_image.size
        header_height, header_image = self._draw_page_hf(
            grid_width, plot_vars, self.config_header
        )
        footer_height, footer_image = self._draw_page_hf(
            grid_width, plot_vars, self.config_footer
        )

        # early exit
        if header_height + footer_height == 0:
            return grid_image

        # create a new image and paste grid and header / footer if applicable
        new_image = self._create_image(
            (grid_width, grid_height + header_height + footer_height),
            self.config_grid.background_color,
        )
        if header_image:
            new_image.paste(header_image, (0, 0))
        new_image.paste(grid_image, (0, header_height))
        if footer_image:
            new_image.paste(footer_image, (0, header_height + grid_height))

        return new_image

    def _draw_page_hf(
        self, width: int, plot_vars: PlotVars, config: PlotConfigHFData = None
    ) -> tuple[int, Image.Image]:
        # early exit
        if not config or (
            not config.text_left and not config.text_center and not config.text_right
        ):
            return (0, None)

        # calc height
        font = self._load_font(config.font, config.font_size)
        texts = [
            self._apply_template(template, plot_vars)
            for template in [config.text_left, config.text_center, config.text_right]
        ]
        height = 0
        for text in texts:
            lines = text.split('\n')
            height = max(height, math.ceil(len(lines) * font.size + 2 * config.padding))

        # create the image
        # if bg color is 'transparent', then keep the same as grid
        image = self._create_image(
            (width, height),
            self.config_grid.background_color
            if config.background_color == 'transparent'
            else config.background_color,
        )

        # calculate the text position
        texts_pos = [
            self._calc_page_hf_text_pos(align, (width, height), config.padding)
            for align in ['left', 'center', 'right']
        ]

        # draw the texts
        draw = ImageDraw.Draw(image)
        draw.font = font
        for i, text in enumerate(texts):
            if not text:
                continue
            # see: https://github.com/python-pillow/Pillow/discussions/7914#discussioncomment-8950499
            # text anchors: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
            draw.multiline_text(
                texts_pos[i][0],
                text,
                font=font,
                fill=config.font_color,
                align=texts_pos[i][1],
                anchor=texts_pos[i][2],
            )

        return (height, image)

    def _apply_template(self, template: str, plot_vars: PlotVars) -> str:
        text = template.replace(r'\n', '\n')
        try:
            text = format_string(text, **asdict(plot_vars))
        except KeyError as e:
            text = "Error: unknown variable '{}'".format(e.args[0])
        return text

    def _calc_page_hf_text_pos(
        self, align: str, size: tuple[int, int], padding: int
    ) -> tuple[tuple[int, int], str, str]:
        # calculate the text position
        match align:
            case 'left':
                pos = (padding, size[1] // 2)
                anchor = 'lm'
            case 'right':
                pos = (size[0] - padding, size[1] // 2)
                anchor = 'rm'
            case 'center':
                pos = (size[0] // 2, size[1] // 2)
                anchor = 'mm'
            case _:
                # probably useless
                raise ValueError(
                    "invalid align value '{}': must be 'left', 'right' or 'center".format(
                        align
                    )
                )
        return (pos, align, anchor)
