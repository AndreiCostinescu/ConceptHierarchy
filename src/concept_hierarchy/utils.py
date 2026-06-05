import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from copy import deepcopy
from typing import Optional, Tuple

tab = "    "


def capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def join_path(*paths) -> str:
    return str(os.path.join(*paths).replace(os.sep, "/"))


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


def read_external_data_content(concept_name, concept_def, concept_hierarchy_file, path_to_root_dir):
    # try to read external file
    external_file = concept_def["data"]
    # first try file relative to the root directory
    if os.path.exists(join_path(path_to_root_dir, external_file)):
        concept_def["data"] = read_json_file(join_path(path_to_root_dir, external_file))
        return
    print("Found external data file:", external_file, "at concept", concept_name, "with wrong format!")
    # then try external files relative to the main file
    external_file_path = join_path("/".join(concept_hierarchy_file.split("/")[:-1]), external_file)
    if os.path.isfile(external_file_path):
        concept_def["data"] = read_json_file(external_file_path)
    elif os.path.isfile(external_file):  # then try external files relative to the script
        concept_def["data"] = read_json_file(external_file)
    else:
        raise RuntimeError("Could not find external data file \"" + external_file + "\" relative to " +
                           "the main file and neither relative to the script for concept {}...".format(concept_name))


def read_concept_hierarchy(concept_hierarchy_file: str, path_to_root_dir: str):
    def create_json(file_name: str) -> json:
        res = read_json_file(file_name)
        if "external" in res:
            assert isinstance(res["external"], list)
            for sub_file_name in res["external"]:
                if os.path.isabs(sub_file_name):
                    res.update(create_json(sub_file_name))
                else:
                    rel_sub_file_name = sanitize_relative_path(join_path(path_to_root_dir, sub_file_name))
                    res.update(create_json(rel_sub_file_name))
            res.pop("external")
        return res

    return create_json(concept_hierarchy_file)


def write_file(file_path, file_content, overwrite_if_same):
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
        same_content = (full_content == existing_content)
    if not same_content or overwrite_if_same:
        print("DEBUG DEBUG DEBUG DEBUG!!!!", ("Creating" if not os.path.exists(file_path) else "Writing"), file_path)
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


def write_json_data(file_path, json_data, json_indent: Optional[int] = 4):
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

    def clone(self) -> 'Graph':
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


def check_ch_name(ch_name: str, name_type: str, must_start_uppercase: bool = False,
                  must_start_lowercase: bool = False, return_bool_instead_of_error: bool = False):
    if ch_name == "":
        if return_bool_instead_of_error:
            return False
        raise SyntaxError("The name of a \"{}\" cannot be empty!".format(name_type))
    illegal_characters_found = []
    for char in ch_name:
        if not (char.isalnum() or char == "_"):
            illegal_characters_found.append(char)
    if ch_name[0].isnumeric():
        if return_bool_instead_of_error:
            return False
        raise SyntaxError("\"{}\" names may not start with a digit! Found in \"{}\"".format(name_type, ch_name))
    if illegal_characters_found:
        if return_bool_instead_of_error:
            return False
        raise SyntaxError("\"{}\" names can not contain the characters ".format(name_type) +
                          ", ".join("'" + c + "'" for c in illegal_characters_found) +
                          ". Found in \"" + ch_name + "\". Delete or replace character by '_'")
    if must_start_lowercase and not ch_name[0].islower():
        if return_bool_instead_of_error:
            return False
        raise SyntaxError("\"{}\" name \"{}\" must start with a lowercase letter!".format(name_type, ch_name))
    if must_start_uppercase and not ch_name[0].isupper():
        if return_bool_instead_of_error:
            return False
        raise SyntaxError("\"{}\" name \"{}\" must start with an uppercase letter!".format(name_type, ch_name))
    return True
