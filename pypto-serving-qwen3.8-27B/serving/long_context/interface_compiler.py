"""Link operator implementations into a serving-owned device forward and cache it."""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory


FAMILY = Path(__file__).with_name("qwen27b")
ABI = "qwen27b-p40b-operator-interface-v2"


def expand_ffn_phases(source, names):
    """Bind FFN phases before PyPTO parsing, preserving the host model's scopes.

    This is compile-time composition, not runtime dispatch. Tensor allocation and
    branch boundaries remain identical to the frozen model. An identity binding
    is required so an operator cannot capture a differently named model local.
    """
    from copy import deepcopy
    tree = ast.parse(source)
    phases = {node.name: node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names}
    if set(phases) != set(names):
        raise ValueError("Missing FFN phase definition")

    class Expand(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in phases:
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in phases:
                phase = phases[value.func.id]
                expected = [ast.Name(arg.arg, ast.Load()) for arg in phase.args.args]
                if value.keywords or ast.dump(ast.List(value.args, ast.Load())) != ast.dump(ast.List(expected, ast.Load())):
                    raise ValueError("FFN phase input binding differs from the serving contract")
                result = phase.body[-1]
                if not isinstance(result, ast.Return) or len(node.targets) != 1:
                    raise ValueError("FFN phase must return its declared output bindings")
                if ast.unparse(node.targets[0]) != ast.unparse(result.value):
                    raise ValueError("FFN phase output bindings differ from the serving contract")
                return deepcopy(phase.body[:-1])
            return self.generic_visit(node)

    return ast.unparse(ast.fix_missing_locations(Expand().visit(tree))) + "\n"


@lru_cache(maxsize=1)
def build_identity():
    """Include compiler/runtime content, including locally patched unversioned builds."""
    result = {}
    for name, paths in {
        "pypto": ("__init__.py", "jit/decorator.py", "ir/compile.py", "ir/distributed_compiled_program.py"),
        "simpler": ("__init__.py", "worker.py"),
    }.items():
        package = importlib.util.find_spec(name)
        directory = Path(package.origin).parent
        selected = [directory / relative for relative in paths]
        if name == "pypto":
            selected.extend(directory.rglob("*.so"))
        for path in sorted(set(selected)):
            if path.is_file():
                result[name + "/" + path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = os.environ.get("PTOAS_ROOT")
    if root:
        for path in (Path(root) / "bin/ptoas", Path(root) / "ptoas"):
            if path.is_file():
                result["ptoas"] = hashlib.sha256(path.read_bytes()).hexdigest()
                break
    return result


def tree_hash(folder):
    records = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(folder.rglob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def validate_operators(fragment, contract, role, kind):
    tree = ast.parse(fragment.source)
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError(f"{role}/{kind} must contain inline operator definitions only")
    definitions = {node.name: node for node in tree.body}
    if set(definitions) != set(contract["symbols"]):
        raise ValueError(f"{role}/{kind} operator symbols do not match the serving ABI")
    for name, signature in contract["abi"].items():
        if ast.unparse(definitions[name].args) != signature:
            raise ValueError(f"{role}/{kind}/{name} has a different operator signature")
    for node in tree.body:
        if not any(ast.unparse(d) == "pl.jit.inline" for d in node.decorator_list):
            raise ValueError(f"{node.name} must be an inline device operator")
        if any(isinstance(n, ast.For) and ast.unparse(n.iter) == "pl.range(64)"
               for n in ast.walk(node)):
            raise ValueError("An operator cannot take over the full model layer loop")


class InterfaceCompiler:
    family = ABI

    def __init__(self, extensions, layout, config):
        if extensions.family != self.family:
            raise ValueError("Unsupported operator model family: " + extensions.family)
        if extensions.family == ABI and (layout.context, layout.page_tokens, layout.head_classes, layout.full_layers, layout.head_dim) != (32768, 128, 4, 16, 256):
            raise ValueError("The current serving model family supports 32K context and its declared head geometry")
        self.extensions, self.layout, self.config = extensions, layout, config
        self.contracts = json.loads((FAMILY / "manifest.json").read_text())
        self.work = TemporaryDirectory(prefix="oneserving-model-")
        self.events = []

    def close(self):
        self.work.cleanup()

    def assemble(self, role):
        contract = self.contracts[role]
        source = f"KV_PHYSICAL_PAGES = {self.layout.physical_pages}\n"
        source += (FAMILY / (role + ".py")).read_text(encoding="utf-8")
        providers = {"attention": self.extensions.attention, "ffn": self.extensions.ffn}
        for kind, expected in contract["operators"].items():
            fragment = providers[kind].bind(role)
            if fragment is None:
                raise ValueError(f"Missing {role}/{kind} implementation")
            validate_operators(fragment, expected, role, kind)
            source += "\n" + fragment.source
        source = expand_ffn_phases(source, contract.get("ffn_phases", []))
        spec = deepcopy(contract["spec"])
        for param in spec["params"]:
            if param["name"].split("__")[0] in ("key_pool", "value_pool"):
                param["shape"] = list(self.layout.pool_shape)
        compile(source, role, "exec")
        return spec, source

    def load(self, role):
        import torch
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
        from .runtime_optimized import DTYPES, logical_shape
        spec, source = self.assemble(role)
        versions = {}
        for name in ("pypto", "torch", "torch-npu"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unpackaged"
        identity = {"abi": self.extensions.family, "source": source, "spec": spec, "versions": versions,
                    "config": str(self.config), "build": build_identity()}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = os.environ.get("ONESERVING_COMPILE_CACHE")
        cache = Path(root).expanduser() / digest if root else None
        module_name = "serving.long_context.qwen27b.bound_" + role + "_" + digest[:12]
        path = Path(self.work.name) / (role + ".py")
        path.write_text(source, encoding="utf-8")
        if cache is not None and cache.is_dir():
            artifact = cache / "program"
            if tree_hash(artifact) != (cache / "sha256").read_text().strip():
                raise ValueError("Corrupt serving compile cache: " + str(cache))
            dest = Path(self.work.name) / (role + "-artifact")
            shutil.copytree(artifact, dest)
            program = DistributedCompiledProgram.from_dir(dest, distributed_config=self.config.distributed_config)
            cached = True
        else:
            descriptor = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(descriptor)
            sys.modules[module_name] = module
            descriptor.loader.exec_module(module)
            samples = [torch.empty(logical_shape(p), dtype=DTYPES[p["dtype"]], device="meta")
                       if p["shape"] is not None else 1 for p in spec["params"]]
            program = getattr(module, spec["entrypoint"]).compile(*samples, config=self.config)
            cached = False
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=cache.parent) as temp:
                    publish = Path(temp) / "published"
                    shutil.copytree(program.output_dir, publish / "program")
                    (publish / "sha256").write_text(tree_hash(publish / "program"))
                    (publish / "identity.json").write_text(json.dumps(
                        {"abi": self.extensions.family, "role": role, "key": digest,
                         "kv_pages": self.layout.physical_pages, "versions": versions}, indent=2))
                    try:
                        publish.rename(cache)
                    except OSError:
                        if not cache.is_dir():
                            raise
        event = {"role": role, "key": digest, "cache_hit": cached,
                 "kv_pages": self.layout.physical_pages, "owner": "serving"}
        self.events.append(event)
        print("[serving-compile] " + json.dumps(event), flush=True)
        return spec, program
