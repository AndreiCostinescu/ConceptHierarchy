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

import json
import os
from collections import defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path

tab = "    "


def capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def join_path(first: str | os.PathLike[str], *rest: str | os.PathLike[str]) -> str:
    return str(os.path.join(first, *rest).replace(os.sep, "/"))


def sanitize_relative_path(path: str) -> str:
    res = []
    res_index = 0
    for part in path.split("/"):
        if part == ".":
            continue
        elif part == ".." and res_index > 0:
            res_index -= 1
            continue
        else:
            if res_index == len(res):
                res.append(part)
            else:
                res[res_index] = part
            res_index += 1
    return "/".join(res[:res_index])


def sanitize_include_relative_paths(include_header: str) -> str:
    include_header_split = include_header.split("<")
    pre = "<".join(include_header_split[:-1]) + ("<" if len(include_header_split) > 0 else "")
    include_header = include_header_split[-1]
    include_header_split = include_header.split(">")
    post = (">" if len(include_header_split) > 0 else "") + ">".join(include_header_split[1:])
    include_header = sanitize_relative_path(include_header_split[0])
    return pre + include_header + post


class Reference:
    def __init__(self, x=None):
        self.ref = x


def is_integer(s, x: Reference = None):
    try:
        if x is None:
            int(s)
        else:
            x.ref = int(s)
        return True
    except ValueError:
        return False


def is_number(s, x: Reference = None):
    try:
        if x is None:
            float(s)
        else:
            x.ref = float(s)
        return True
    except ValueError:
        return False


def read_json_file(file_name: str):
    with open(file_name) as f:
        res = json.load(f)
    return res


def read_external_data_content(concept_name, external_file, concept_hierarchy_file, path_to_root_dir) -> object:
    # try to read external file
    # first try file relative to the root directory
    file_path = join_path(path_to_root_dir, external_file)
    if os.path.isfile(file_path):
        return read_json_file(file_path)
    print(f"Found external data file: {external_file!r} at concept {concept_name} with wrong format!")
    # then try external files relative to the main file
    external_file_path = join_path("/".join(concept_hierarchy_file.split("/")[:-1]), external_file)
    if os.path.isfile(external_file_path):
        return read_json_file(external_file_path)
    elif os.path.isfile(external_file):  # then try external files relative to the python script
        return read_json_file(external_file)
    else:
        raise RuntimeError(
            f"Could not find external data file {external_file!r} relative to the main file and neither relative to the"
            f" script at concept {concept_name}..."
        )


def write_file(file_path, file_content, overwrite_if_same, just_testing: bool = False):
    if isinstance(file_content, str):
        full_content = file_content
    else:
        full_content = "\n".join(file_content)
    if not full_content.endswith("\n"):
        full_content += "\n"
    same_content = False
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_content = "".join(f.readlines())
        """
        print("vs")
        print(full_content)
        print("vs")
        print(existing_content)
        print("vs")
        """
        same_content = full_content == existing_content
    if not same_content or overwrite_if_same:
        print("DEBUG DEBUG DEBUG DEBUG!!!!", ("Creating" if not os.path.exists(file_path) else "Writing"), file_path)
        if just_testing:
            return
        file_path_directory = "" if "/" not in file_path else "/".join(file_path.split("/")[:-1])
        if file_path_directory:
            Path(file_path_directory).mkdir(parents=True, exist_ok=True)  # creates the path if it doesn't exist
        with open(file_path, "w") as f:
            f.write(full_content)


def get_today_string() -> str:
    return date.today().strftime("%d.%m.%y.")


def get_file_timestamp(file_path, overwrite_timestamp, current_timestamp_string=None):
    if current_timestamp_string is None:
        current_timestamp_string = "// Created by Andrei on " + get_today_string()
    if not overwrite_timestamp and os.path.exists(file_path):
        with open(file_path, "r") as f:
            file_content = f.readlines()
            if len(file_content) >= 3 and file_content[1].startswith("// "):
                return file_content[1].strip()
            return current_timestamp_string
    return current_timestamp_string


def write_json_data(file_path, json_data, json_indent: int | None = 4):
    write_file(file_path, json.dumps(json_data, indent=json_indent), False)


# Inspired by Neelam Yadav (https://www.geeksforgeeks.org/python-program-for-topological-sorting/)
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.nodes = {}

    # function to add an edge to graph
    def add_edge(self, u, v):
        if u not in self.nodes:
            self.nodes[u] = True
        if v not in self.nodes:
            self.nodes[v] = True
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topological_sort_util(self, v, visited, stack):
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive topological_sort_util()
    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = {x: False for x in self.nodes}
        stack = []

        # Call the recursive helper function to store Topological Sort starting from all vertices one by one
        for i in self.nodes:
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        # Print contents of stack
        return stack

    def clone(self) -> "Graph":
        res = Graph()
        res.graph = deepcopy(self.graph)
        res.nodes = deepcopy(self.nodes)
        return res


def get_items_of_single_entry_dict(d: dict) -> tuple:
    assert len(d) == 1
    for key, value in d.items():
        return key, value
    raise RuntimeError("Dictionary is empty... can't get single-entry!")


def replace_template_chars(x: str) -> str:
    return x.replace("<", "__").replace(">", "").replace(", ", "_").replace("!", "not")


# Topological sort (Kahn's algorithm)
def topological_sort(parents: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    from collections import deque

    # children[parent] = list of child names
    children: dict[str, list[str]] = {name: [] for name in parents}
    in_degree: dict[str, int] = {name: 0 for name in parents}

    for c, c_parents in parents.items():
        for parent in c_parents:
            children[parent].append(c)
            in_degree[c] += 1

    roots: list[str] = [name for name, degree in in_degree.items() if degree == 0]
    queue: deque[str] = deque(roots)
    result: list[str] = []

    while queue:
        name = queue.popleft()
        result.append(name)
        for child in children[name]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(parents):
        nodes_in_cycles = set(parents) - set(result)
        raise RuntimeError(
            f"Non-hierarchy structure detected! The following items form one or more cycles: {nodes_in_cycles!r}"
        )
    return result, roots
