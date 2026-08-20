import unittest

from src.python.shared.formatting import format_string


class FormatStringExpressionTests(unittest.TestCase):
    def test_slice_expression(self):
        self.assertEqual(format_string('{dim1[:5]}', dim1='abcdef'), 'abcde')
        self.assertEqual(format_string('{dim1[2:]}', dim1='abcdef'), 'cdef')

    def test_replace_expression_accepts_windows_paths(self):
        self.assertEqual(
            format_string(
                '{dim1.replace("J:/models/loras/", "")}',
                dim1='J:/models/loras/epoch_04.safetensors',
            ),
            'epoch_04.safetensors',
        )

    def test_prefix_suffix_and_existing_formats(self):
        self.assertEqual(
            format_string('{dim1.removeprefix("epoch_")}', dim1='epoch_04'),
            '04',
        )
        self.assertEqual(
            format_string('{dim1.removesuffix(".safetensors")}', dim1='epoch.safetensors'),
            'epoch',
        )
        self.assertEqual(format_string('{dim2[1]:g}', dim2=('weight', 1.4)), '1.4')

    def test_unsupported_expression_is_rejected(self):
        with self.assertRaises(ValueError):
            format_string('{dim1.replace("old")}', dim1='old')


if __name__ == '__main__':
    unittest.main()
