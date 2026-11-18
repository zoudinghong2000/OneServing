"""Bind document operators at the native Qwen prefill layer boundaries."""
import ast
import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

from .interface_compiler import InterfaceCompiler, expand_ffn_phases, validate_operators

FAMILY = "qwen27b-native-pages-v1"
KERNELS = Path(__file__).with_name("kernels")
BUCKETS = {
    "capture": "prefill_chunk512",
    "hit_causal": "prefill_chunk512",
    "wide_causal": "prefill_chunk512",
    "tail16": "prefill_small16",
    "tail64": "prefill_wide64",
}


@dataclass(frozen=True)
class NativePageLayout:
    pool_shape: tuple[int, ...]

    @property
    def physical_pages(self):
        return self.pool_shape[2]

    @classmethod
    def from_decode(cls, specs):
        params = specs["decode_greedy"]["params"]
        pools = [tuple(p["shape"]) for p in params if p["name"].split("__")[0] in ("key_pool", "value_pool")]
        if len(pools) != 2 or pools[0] != pools[1] or len(pools[0]) != 5:
            raise ValueError("Native K and V must share the layer/head/page/token/channel layout")
        return cls(pools[0])

    @classmethod
    def for_model(cls):
        return cls.from_decode(json.loads((KERNELS / "graph_specs.json").read_text()))


def bind_phases(source, functions):
    """Replace only explicitly marked FFN sections in the native template."""
    bound = []

    def bind(match):
        indent, name = match.group("indent", "name")
        bound.append(name)
        phase = functions["redknot_ffn_" + name]
        args = ", ".join(arg.arg for arg in phase.args.args)
        return f"{indent}{ast.unparse(phase.body[-1].value)} = {phase.name}({args})"

    result = re.sub(r"^(?P<indent> +)# operator: (?P<name>\w+)\n.*?^(?P=indent)# end operator: (?P=name)$",
                  bind, source, flags=re.MULTILINE | re.DOTALL)
    if bound != ["initialize", "prepare", "prepare", "execute"]:
        raise ValueError("Native prefill FFN boundaries differ from the backend contract")
    return result


def extra_args(params):
    types = {"int32": "INT32", "bfloat16": "BF16", "fp32": "FP32"}
    result = []
    for param in params:
        name = param["name"].split("__")[0]
        dtype = "pl." + types[param["dtype"]]
        if param["shape"] is None:
            annotation = f"pl.Scalar[{dtype}]"
        else:
            annotation = f"pl.Tensor[{param['shape']}, {dtype}]"
            if param["direction"] == "InOut":
                annotation = f"pl.InOut[{annotation}]"
        result.append(ast.arg(arg=name, annotation=ast.parse(annotation, mode="eval").body))
    return result


def specialize(source, role, params, functions):
    tail = role.startswith("tail")
    maskless = role.endswith("_causal")
    if not tail:
        source = bind_phases(source, functions)
    module = ast.parse(source)
    extras = extra_args(params)
    wrapper_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)
                     and (node.name.startswith("prefill_batch_") or node.name in ("rank_prefill_shared", "prefill_fwd"))}

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wrapper_names:
            continue
        is_body = any(isinstance(stmt, ast.For) for stmt in node.body)
        if maskless:
            node.args.args = [a for a in node.args.args if a.arg != "mask"]
        node.args.args.extend(deepcopy(extras))
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            name = call.func.id
            if name in wrapper_names:
                if maskless:
                    call.args = [arg for arg in call.args if not isinstance(arg, ast.Name) or arg.id != "mask"]
                call.args.extend(ast.Name(a.arg, ast.Load()) for a in extras)
            elif name.startswith("prefill_attention_slots_inline"):
                call.args.extend(ast.Name(a.arg, ast.Load()) for a in extras[:5])
            elif name.startswith("prefill_delta_slots_inline") and not tail:
                call.args.extend(ast.parse("f(rk_windows, rk_core, rk_state, rk_start, rk_end, rk_mode, "
                                           "pl.read(cache_offsets, [0]))").body[0].value.args)
        if is_body and maskless:
            node.body[:0] = [ast.parse("mask = boundary_mask(cache_offsets, token_counts)").body[0]]

    if role == "wide_causal":
        # Within the entry's layer/embedding loop 512 denotes the row bucket.
        # Projection task counts are row tiles, distinct from weight tile dimensions.
        for node in module.body:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "ROWS":
                node.value = ast.Constant(1024)
            if isinstance(node, ast.FunctionDef) and node.name in wrapper_names:
                for constant in ast.walk(node):
                    if isinstance(constant, ast.Constant) and constant.value == 512:
                        constant.value = 1024
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("prefill_project_inline_prefix512",
                                                                  "prefill_project_ffn_up_paired_m128"):
                blocks = 4 if node.name == "prefill_project_inline_prefix512" else 2
                for child in ast.walk(node):
                    if isinstance(child, ast.BinOp) and isinstance(child.left, ast.Name) and child.left.id == "task" and isinstance(child.op, (ast.FloorDiv, ast.Mod)):
                        child.right = ast.Constant(blocks * 2)
                    if isinstance(child, ast.BinOp) and isinstance(child.right, ast.Name) and child.right.id == "output_tiles" and isinstance(child.op, ast.Mult):
                        child.left = ast.Constant(blocks * 2)

    if maskless:
        mask = Path(__file__).with_name("qwen27b") / "boundary_mask.py"
        functions["boundary_mask"] = ast.parse(mask.read_text()).body[0]
    module.body = [node for node in module.body if not isinstance(node, ast.FunctionDef)
                   or node.name not in functions]
    module.body += list(functions.values())
    return ast.unparse(ast.fix_missing_locations(module))


class NativePageCompiler(InterfaceCompiler):
    family = FAMILY

    def assemble(self, role):
        contract = self.contracts[role]
        # Use the standard entry and the same resident tensor shapes as decode.
        spec = deepcopy(contract["spec"])
        extra = []
        for param in spec["params"]:
            name = param["name"].split("__")[0]
            if name in ("key_pool", "value_pool"):
                param["shape"] = list(self.layout.pool_shape)
            elif name.startswith("rk_"):
                extra.append(param)
        functions = {}
        for kind, expected in contract["operators"].items():
            fragment = getattr(self.extensions, kind).bind(role)
            validate_operators(fragment, expected, role, kind)
            functions.update((node.name, node) for node in ast.parse(fragment.source).body)
        source = (KERNELS / (BUCKETS[role] + ".py")).read_text(encoding="utf-8")
        source = specialize(source, role, extra, functions)
        source = expand_ffn_phases(source, contract.get("ffn_phases", ()))
        # The model module supplies inline operators as well as constant dimensions.
        model_hash = hashlib.sha256((KERNELS.parent / "model.py").read_bytes()).hexdigest()
        source += f"\n# Native model dependency {model_hash}\n"
        compile(source, role, "exec")
        return spec, source
