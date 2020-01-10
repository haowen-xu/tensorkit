import codecs
import os
import re
from typing import *


fast_dtype_names = ('float32', 'int32')
dtype_names = ('int8', 'uint8', 'int16', 'int64', 'float16', 'float64', 'bool')


def make_dtype_mapper(rhs,
                      indent: str = '    ',
                      directly_return: bool = False,
                      target_prefix: str = 'target_',
                      dtype_filter: Callable[[str], bool] = (lambda s: True),
                      reverse: bool = False):
    lhs = f'{target_prefix}{rhs} =' if not directly_return else 'return'
    f = lambda k: f"'{k}'"
    g = lambda k: f'torch.{k}'
    if reverse:
        f, g = g, f

    ret = (
        indent +
        f'{lhs} ' +
        '{' +
        (', '.join(f"{f(k)}: {g(k)}" for k in dtype_names if dtype_filter(k))) +
        '}' + f'[{rhs}]'
    )

    if fast_dtype_names:
        buf = []
        i = 0
        for k in fast_dtype_names:
            if dtype_filter(k):
                buf.append(
                    indent + f"{'el' if i > 0 else ''}if {rhs} == {f(k)}:\n" +
                    indent + f"    {lhs} {g(k)}\n"
                )
                i += 1
        if buf:
            buf.append(indent + f'else:\n    {ret}')
            ret = ''.join(buf)

    return ret


def replace_dtype_mapper(cnt):
    def f(m):
        return make_dtype_mapper('dtype', m.group(1))

    cnt = re.sub(
        r"^([ ]+)(?:"
        r"if dtype == '[^']+':(?:(?!def).)*?else:\s+(?:target_)?dtype = \{[^{}]*\}\[dtype\]|"
        r"(?!<if dtype == .*?:\s+)(?:target_)?dtype = \{[^{}]*\}\[dtype\]"
        r")",
        f,
        cnt,
        flags=re.MULTILINE | re.DOTALL
    )

    def f(m):
        return make_dtype_mapper('dtype', m.group(1), target_prefix='real_',
                                 dtype_filter=(lambda s: s.startswith('float')))

    cnt = re.sub(
        r"^([ ]+)(?:"
        r"if dtype == '[^']+':(?:(?!def).)*?else:\s+real_dtype = \{[^{}]*\}\[dtype\]|"
        r"(?!<if dtype == .*?:\s+)real_dtype = \{[^{}]*\}\[dtype\]"
        r")",
        f,
        cnt,
        flags=re.MULTILINE | re.DOTALL
    )

    def f(m):
        return make_dtype_mapper('dtype', m.group(1), target_prefix='int_',
                                 dtype_filter=(lambda s: s.startswith('int')))

    cnt = re.sub(
        r"^([ ]+)(?:"
        r"if dtype == '[^']+':(?:(?!def).)*?else:\s+int_dtype = \{[^{}]*\}\[dtype\]|"
        r"(?!<if dtype == .*?:\s+)int_dtype = \{[^{}]*\}\[dtype\]"
        r")",
        f,
        cnt,
        flags=re.MULTILINE | re.DOTALL
    )

    return cnt


def replace_reverse_dtype_mapper(cnt):
    def f(m):
        return make_dtype_mapper(m.group(2), m.group(1), directly_return=True,
                                 reverse=True)

    return re.sub(
        r"^([ ]+)"
        r"if ([a-z]+\.dtype) == torch\.(?:(?!def).)*?else:\s+return \{[^{}]*\}\[\2\]",
        f,
        cnt,
        flags=re.MULTILINE | re.DOTALL
    )


def main():
    parent_dir = os.path.split(os.path.abspath(__file__))[0]
    for name in os.listdir(parent_dir):
        if name.endswith('.py'):
            file_path = os.path.join(parent_dir, name)
            if os.path.samefile(file_path, __file__):
                continue

            with codecs.open(file_path, 'rb', 'utf-8') as f:
                cnt = f.read()
            cnt2 = replace_dtype_mapper(cnt)
            cnt2 = replace_reverse_dtype_mapper(cnt2)

            if cnt2 != cnt:
                print(f'updated: {name}')
                with codecs.open(file_path, 'wb', 'utf-8') as f:
                    f.write(cnt2)


if __name__ == '__main__':
    main()

