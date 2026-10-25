import json
from candycrunch.cli import build_parser, main

# The in-browser CandyCrunch (Pyodide in a web worker): the page builds its settings form from parameters() and runs the CLI with run() on the files it
# mounted, reading the output files from Pyodide's file system


def parameters():
    """the CLI's options as JSON, for the page's settings form\n
   | Returns:
   | :-
   | Returns a JSON list with one object per option: name, flag, help, type (str, int, float or bool), choices, nargs, default and required
   """
    types = {'str_to_bool': 'bool'}
    return json.dumps([{'name': a.dest, 'flag': a.option_strings[0], 'help': a.help, 'type': types.get(getattr(a.type, '__name__', None), getattr(a.type, '__name__', None)),
                        'choices': a.choices, 'nargs': a.nargs, 'default': a.default, 'required': a.required} for a in build_parser()._actions if a.dest != 'help'])


def run(argv):
    """runs the CLI as from the command line\n
   | Arguments:
   | :-
   | argv (list): the options as strings, --spectra_filepath the mounted files and --output a path in Pyodide's file system\n
   | Returns:
   | :-
   | Nothing; the CLI writes its tables to --output (several files: also one per file, named after it). An invalid option raises ValueError instead of
   | ending the interpreter, its message printed to stderr (the page's log) by the CLI
   """
    try:
        main(argv)
    except SystemExit as e:
        if e.code:
            raise ValueError(f'invalid options (exit code {e.code}), see the message above') from None
