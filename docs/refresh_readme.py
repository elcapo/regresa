from string import Template
from os.path import dirname, join
from regresa import linear, logistic

exclusions = ['math', 'np', 'numpyize_list']

script_dir = dirname(__file__)
readme_template_path = join(script_dir, 'README.template')
readme_path = join(script_dir, '..', 'README.md')

with open(readme_template_path, 'r') as f:
    readme_template = Template(f.read())

docs = {}

for name in dir(linear):
    function = eval('linear.' + name)
    doc = getattr(function, '__doc__')
    if not name in exclusions and not name.startswith('__'):
        docs['linear_' + name + '_help'] = doc

for name in dir(logistic):
    function = eval('logistic.' + name)
    doc = getattr(function, '__doc__')
    if not name in exclusions and not name.startswith('__'):
        docs['logistic_' + name + '_help'] = doc

with open(readme_path, 'w') as f:
    f.write(readme_template.safe_substitute(docs))
