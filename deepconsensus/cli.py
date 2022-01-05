# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# pylint: disable=g-short-docstring-punctuation
"""DeepConsensus

Usage:
  deepconsensus <command> [optional arguments]

Commands:
  preprocess: Convert aligned subreads to tf.Example format.
  run: Run DeepConsenseus beginning with aligned subreads.

"""

import argparse
import sys
import textwrap

from absl import app
from absl.flags import argparse_flags

from deepconsensus.utils import dc_constants

COMMANDS = ['preprocess', 'run']


def parse_flags(argv):
  parser = argparse_flags.ArgumentParser(add_help=False, usage=__doc__)
  parser.add_argument('command', choices=COMMANDS, help=argparse.SUPPRESS)
  parser.add_argument(
      '--version', action='version', version=dc_constants.__version__)
  return parser.parse_known_args(argv[1:])


def handle_help(passed, module):
  """Print a better help screen for subcommands."""
  if '-h' in passed or '--help' in passed or len(sys.argv) == 2:
    flag_set = module.flags.FLAGS.flags_by_module_dict()[module.__name__]
    print(module.__doc__, file=sys.stderr)
    flag_help = []
    for flag in flag_set:
      out = f'    --{flag.name:<20} {flag.help}'
      if flag.default:
        out += f' [default: {flag.default}]'
      flag_help.append(out)
    print('Flags:', file=sys.stderr)
    for flag in flag_help:
      print(
          textwrap.fill(
              flag,
              width=80,
              subsequent_indent=' ' * 27,
              fix_sentence_endings=True),
          file=sys.stderr)
    print('', file=sys.stderr)
    print('Requirements:', file=sys.stderr)
    for flag in flag_set:
      for validator in flag.validators:
        print('    ' + validator.message, file=sys.stderr)
    # Print help and exit.
    exit(0)


def main(argset):
  args, passed = argset  # Ignore unused args; These are passed to subcommands.
  if args.command:
    passed = [args.command] + passed
  if args.command == 'preprocess':
    from deepconsensus.preprocess import preprocess
    preprocess.register_required_flags()
    handle_help(passed, preprocess)
    app.run(preprocess.main, argv=passed)
  elif args.command == 'run':
    from deepconsensus.inference import quick_inference
    quick_inference.register_required_flags()
    handle_help(passed, quick_inference)
    app.run(quick_inference.main, argv=passed)


def run():
  app.run(main, flags_parser=parse_flags)


if __name__ == '__main__':
  run()
