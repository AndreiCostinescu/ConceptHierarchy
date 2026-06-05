# Copyright 2026 ConceptHierarchy Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
cli.py — Command-line interface for the ConceptHierarchy compiler.

Usage
-----
    concept-hierarchy compile my_hierarchy.json --target cpp --output out.hpp
    concept-hierarchy --version
"""

from __future__ import annotations

import argparse
import sys

from concept_hierarchy import __version__
from concept_hierarchy.compiler import compile_hierarchy
from concept_hierarchy.errors import ConceptHierarchyError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concept-hierarchy",
        description="Compiler for the ConceptHierarchy programming language.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    sub = parser.add_subparsers(dest="command")

    # ---- compile ----------------------------------------------------------
    compile_p = sub.add_parser(
        "compile", help="Compile a ConceptHierarchy JSON definition."
    )
    compile_p.add_argument(
        "source",
        help="Path to the ConceptHierarchy JSON file.",
    )
    compile_p.add_argument(
        "--target",
        default="cpp",
        choices=["cpp"],
        help="Target language for code generation (default: cpp).",
    )
    compile_p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write generated code to FILE (default: stdout).",
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "compile":
        try:
            code = compile_hierarchy(
                args.source,
                target=args.target,
                output_path=args.output,
            )
            if args.output is None:
                print(code)
        except ConceptHierarchyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
