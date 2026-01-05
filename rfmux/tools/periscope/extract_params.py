''' Class to extract parameters from the existing dialog and compare it with the default test'''

import inspect
import ast
from importlib import import_module

class ParamKeyExtractor:
    """Extracts parameter keys used inside `<class>.get_parameters()`."""

    PARAM_CONTAINERS = {"params", "params_dict"}

    def __init__(self, module_path: str, class_name: str):
        self.module_path = module_path
        self.class_name = class_name
        self.tree = self._parse_module_ast()

    # ----------------------------
    # Parsing + Class/Method Lookup
    # ----------------------------

    def _parse_module_ast(self) -> ast.Module:
        try:
            module = import_module(self.module_path)
            source = inspect.getsource(module)
            return ast.parse(source)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Unable to parse {self.module_path} for parameter key validation: {exc}"
            ) from exc

    def _find_class(self) -> ast.ClassDef | None:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == self.class_name:
                return node
        return None

    def _find_method(self, class_node: ast.ClassDef) -> ast.FunctionDef | None:
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "get_parameters":
                return node
        return None

    # ----------------------------
    # Extraction Rules
    # ----------------------------

    def _extract_keys_from_dict(self, node: ast.Dict) -> set[str]:
        return {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

    def _extract_keys_from_assignment(self, node: ast.Assign) -> set[str]:
        """Handles patterns like: params.foo = value"""
        keys = set()
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.PARAM_CONTAINERS
            ):
                keys.add(target.attr)
        return keys

    def _extract_keys_from_subscript(self, node: ast.Subscript) -> set[str]:
        """Handles patterns like: params['foo']"""
        if isinstance(node.value, ast.Name) and node.value.id in self.PARAM_CONTAINERS:
            slice_node = node.slice
            # Python <3.9 compatibility
            if isinstance(slice_node, ast.Index):
                slice_node = slice_node.value
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                return {slice_node.value}
        return set()


    # ----------------------------
    # Main Extraction
    # ----------------------------

    def extract(self) -> set[str]:
        """Main public method: returns all parameter keys found."""
        class_node = self._find_class()
        if not class_node:
            return set()

        method_node = self._find_method(class_node)
        if not method_node:
            return set()

        keys: set[str] = set()

        for node in ast.walk(method_node):
            if isinstance(node, ast.Dict):
                keys.update(self._extract_keys_from_dict(node))
            elif isinstance(node, ast.Assign):
                keys.update(self._extract_keys_from_assignment(node))
            elif isinstance(node, ast.Subscript):
                keys.update(self._extract_keys_from_subscript(node))

        return keys
