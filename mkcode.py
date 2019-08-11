import codecs
import os
from glob import glob

from jinja2 import Environment, FileSystemLoader


def format_all_list(val):
    lines = ['']
    for v in val:
        v_lit = repr(v) + ','
        if len(v_lit) + len(lines[-1]) > 75:
            lines.append('    ')
        lines[-1] += v_lit + ' '
    for i, line in enumerate(lines):
        lines[i] = line.rstrip()
    if not lines[-1]:
        lines.pop(-1)
    lines[-1] = lines[-1].rstrip(',')
    return '\n'.join(lines)


for src in glob('**/*.pyt', recursive=True):
    dst = os.path.splitext(src)[0] + '.py'
    env = Environment(
        loader=FileSystemLoader(
            searchpath=[
                os.path.split(os.path.abspath(src))[0],
                '.'
            ]
        )
    )
    t = env.get_template(src)
    rendered = t.render(format_all_list=format_all_list)
    rendered = rendered.rstrip() + '\n'
    old = None
    if os.path.isfile(dst):
        with codecs.open(dst, 'rb', 'utf-8') as f:
            old = f.read()
    if old != rendered:
        with codecs.open(dst, 'wb', 'utf-8') as f:
            f.write(rendered)
