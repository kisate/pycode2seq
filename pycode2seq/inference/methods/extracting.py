from pycode2seq.inference.methods.model import MethodInfo
from pycode2seq.inference.common.utils import pre_order, set_node_technical_token, split_to_subtokens


def extract_label(method_info: MethodInfo, hide_method_names: bool) -> str:
    if method_info.method.name_node is None:
        return None
    if method_info.name is None:
        return None

    method_name_node = method_info.method.name_node
    method_root = method_info.method.root
    method_name = method_info.name

    if hide_method_names:
        for node in pre_order(method_root):
            if node.token == method_name:
                set_node_technical_token(node, "SELF")

        set_node_technical_token(method_name_node, "METHOD_NAME")

    return "|".join(split_to_subtokens(method_name))
