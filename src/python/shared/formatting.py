"""Safe extensions for the Python format-string mini-language.

The standard ``str.format`` syntax is retained.  This formatter additionally
supports a small set of string expressions inside replacement fields, for
example ``{dim1[:20]}`` or
``{dim1.replace('C:/models/loras/', '')}``.

Expressions are deliberately restricted to indexing, slicing, and a small
allow-list of string methods.  They are not evaluated as arbitrary Python.
"""

import ast
import re
import string
from typing import Any


_POSITIONAL_ROOT = re.compile(r'^(\d+)(.*)$')
_EXPRESSION_METHODS = {
    'replace': (2, 3),
    'removeprefix': (1, 1),
    'removesuffix': (1, 1),
    'strip': (0, 1),
    'lstrip': (0, 1),
    'rstrip': (0, 1),
    'lower': (0, 0),
    'upper': (0, 0),
}


class _SafeFormatter(string.Formatter):
    @staticmethod
    def _normalize_positional_root(field_name: str) -> str:
        match = _POSITIONAL_ROOT.match(field_name)
        if not match or not match.group(2):
            return field_name
        return f'arg{match.group(1)}{match.group(2)}'

    def _evaluate(self, node: ast.AST, args, kwargs) -> Any:
        if isinstance(node, ast.Name):
            if node.id in kwargs:
                return kwargs[node.id]
            if node.id.startswith('arg') and node.id[3:].isdigit():
                index = int(node.id[3:])
                return args[index]
            raise KeyError(node.id)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            return node.value

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._evaluate(node.operand, args, kwargs)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError('unary signs are only allowed on numbers')
            return -value if isinstance(node.op, ast.USub) else value

        if isinstance(node, ast.Subscript):
            value = self._evaluate(node.value, args, kwargs)
            key = self._evaluate_slice(node.slice, args, kwargs)
            return value[key]

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = self._evaluate(node.func.value, args, kwargs)
            method = node.func.attr
            if not isinstance(value, str) or method not in _EXPRESSION_METHODS:
                raise ValueError(f"string method '{method}' is not allowed")
            if node.keywords:
                raise ValueError('keyword arguments are not allowed')
            method_args = [self._evaluate(argument, args, kwargs) for argument in node.args]
            min_args, max_args = _EXPRESSION_METHODS[method]
            if not min_args <= len(method_args) <= max_args:
                raise ValueError(
                    f"string method '{method}' expects {min_args} to {max_args} arguments"
                )
            if method in {'replace', 'removeprefix', 'removesuffix', 'strip', 'lstrip', 'rstrip'}:
                if not all(isinstance(argument, str) for argument in method_args[:2]):
                    raise TypeError(f"string method '{method}' expects string arguments")
            return getattr(value, method)(*method_args)

        raise ValueError('only indexing, slicing, and approved string methods are allowed')

    def _evaluate_slice(self, node: ast.AST, args, kwargs):
        if isinstance(node, ast.Slice):
            return slice(
                self._evaluate_optional(node.lower, args, kwargs),
                self._evaluate_optional(node.upper, args, kwargs),
                self._evaluate_optional(node.step, args, kwargs),
            )
        return self._evaluate(node, args, kwargs)

    def _evaluate_optional(self, node: ast.AST | None, args, kwargs):
        return None if node is None else self._evaluate(node, args, kwargs)

    def vformat(self, format_string, args, kwargs):
        rewritten, expressions = _rewrite_expression_fields(format_string)
        if not expressions:
            return super().vformat(format_string, args, kwargs)

        # Put lazy values into a private copy so the normal Formatter can still
        # handle alignment, numeric formats, conversions, and escaped braces.
        format_kwargs = dict(kwargs)
        for name, expression in expressions.items():
            format_kwargs[name] = _LazyExpression(self, expression, args, format_kwargs)
        return super().vformat(rewritten, args, format_kwargs)


class _LazyExpression:
    def __init__(self, formatter, expression, args, kwargs):
        self.formatter = formatter
        self.expression = expression
        self.args = args
        self.kwargs = kwargs

    def __format__(self, format_spec):
        try:
            expression = self.formatter._normalize_positional_root(self.expression)
            tree = ast.parse(expression, mode='eval').body
            value = self.formatter._evaluate(tree, self.args, self.kwargs)
            return format(value, format_spec)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, SyntaxError) as error:
            raise ValueError(
                f"Invalid format expression '{{{self.expression}}}': {error}"
            ) from error


def _rewrite_expression_fields(template: str) -> tuple[str, dict[str, str]]:
    """Replace extended fields with normal fields before Formatter parses them.

    Python's Formatter treats every top-level colon as the start of a format
    specifier.  Scanning the field ourselves lets expressions contain colons
    inside slices and quoted Windows paths.
    """
    output = []
    expressions = {}
    index = 0
    expression_index = 0
    while index < len(template):
        if template.startswith('{{', index) or template.startswith('}}', index):
            output.append(template[index : index + 2])
            index += 2
            continue
        if template[index] != '{':
            output.append(template[index])
            index += 1
            continue

        end = _find_field_end(template, index + 1)
        if end is None:
            output.append(template[index])
            index += 1
            continue

        body = template[index + 1 : end]
        expression, format_spec = _split_expression_format_spec(body)
        if _is_extended_expression(expression):
            name = f'__comfylab_expression_{expression_index}'
            expression_index += 1
            expressions[name] = expression
            output.append('{')
            output.append(name)
            if format_spec:
                output.extend((':', format_spec))
            output.append('}')
        else:
            output.append(template[index : end + 1])
        index = end + 1
    return ''.join(output), expressions


def _is_extended_expression(expression: str) -> bool:
    if '(' in expression:
        return True
    bracket_depth = 0
    quote = None
    escaped = False
    for character in expression:
        if quote:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == quote:
                quote = None
        elif character in "'\"":
            quote = character
        elif character == '[':
            bracket_depth += 1
        elif character == ']':
            bracket_depth -= 1
        elif character == ':' and bracket_depth:
            return True
    return False


def _find_field_end(template: str, start: int) -> int | None:
    quote = None
    escaped = False
    nesting = 0
    for index in range(start, len(template)):
        character = template[index]
        if quote:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
        elif character in '[(':
            nesting += 1
        elif character in '])':
            nesting -= 1
        elif character == '{':
            nesting += 1
        elif character == '}' and nesting == 0:
            return index
        elif character == '}':
            nesting -= 1
    return None


def _split_expression_format_spec(body: str) -> tuple[str, str]:
    quote = None
    escaped = False
    nesting = 0
    for index, character in enumerate(body):
        if quote:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
        elif character in '[(':
            nesting += 1
        elif character in '])':
            nesting -= 1
        elif character == ':' and nesting == 0:
            return body[:index], body[index + 1 :]
    return body, ''


_FORMATTER = _SafeFormatter()


def format_string(template: str, *args, **kwargs) -> str:
    """Format a template using standard formatting plus safe string expressions."""
    return _FORMATTER.vformat(template, args, kwargs)
