from typing import List
from pycode2seq.inference.common.utils import pre_order
from pycode2seq.inference.common.node import Node

from pycode2seq.inference.methods.model import ElementNode, MethodInfo, MethodNode, ParameterNode

from pycode2seq.inference.parsing.utils import decompress_type_label

METHOD_NODE = "methodDeclaration"
METHOD_RETURN_TYPE_NODE = "typeTypeOrVoid"
METHOD_NAME_NODE = "IDENTIFIER"

CLASS_DECLARATION_NODE = "classDeclaration"
CLASS_NAME_NODE = "IDENTIFIER"

METHOD_PARAMETER_NODE = "formalParameters"
METHOD_PARAMETER_INNER_NODE = "formalParameterList"
METHOD_SINGLE_PARAMETER_NODE = ["formalParameter", "lastFormalParameter"]
PARAMETER_RETURN_TYPE_NODE = "typeType"
PARAMETER_NAME_NODE = "variableDeclaratorId"


def split_java_into_methods(root: Node) -> List[MethodInfo]:
    method_roots = [node for node in pre_order(root) if decompress_type_label(node.type_label)[-1] == METHOD_NODE]
    return [collect_method_info(root) for root in method_roots]

def collect_method_info(method_node: Node) -> MethodInfo:
    method_name = method_node.get_child_of_type(METHOD_NAME_NODE)
    method_return_type_node = method_node.get_child_of_type(METHOD_RETURN_TYPE_NODE)
    method_return_type_node.token = collect_parameter_token(method_return_type_node)

    class_root = get_enclosing_class(method_node)
    class_name = class_root.get_child_of_type(CLASS_NAME_NODE) if class_root is not None else None

    parameters_root = method_node.get_child_of_type(METHOD_PARAMETER_NODE)
    inner_parameters_root = parameters_root.get_child_of_type(METHOD_PARAMETER_INNER_NODE)

    if inner_parameters_root is not None:
        parameters_list = get_list_of_parameters(inner_parameters_root)
    elif parameters_root is not None:
        parameters_list = get_list_of_parameters(parameters_root)
    else:
        parameters_list = []

    return MethodInfo(
        MethodNode(method_node, method_return_type_node, method_name),
        ElementNode(class_root, class_name),
        parameters_list
    )

def get_enclosing_class(root: Node) -> Node:
    if decompress_type_label(root.type_label)[-1] == CLASS_DECLARATION_NODE:
        return root

    parent = root.parent
    if (parent is not None):
        return get_enclosing_class(parent)

    return None

def get_list_of_parameters(root: Node) -> List[ParameterNode]:
    if decompress_type_label(root.type_label)[-1] == METHOD_SINGLE_PARAMETER_NODE:
        return [get_parameter_info_from_node(root)]
    
    return [
        get_parameter_info_from_node(child) for child in root.children 
        if decompress_type_label(child.type_label)[0] in METHOD_SINGLE_PARAMETER_NODE
    ]
    

def get_parameter_info_from_node(root: Node) -> ParameterNode:
    return_type_node = root.get_child_of_type(PARAMETER_RETURN_TYPE_NODE)
    return_type_node.token = collect_parameter_token(return_type_node)

    return ParameterNode(
        root,
        return_type_node,
        root.get_child_of_type(PARAMETER_NAME_NODE)
    )

def collect_parameter_token(root: Node) -> str:
    if (root.is_leaf()):
        return root.token

    return "".join([collect_parameter_token(child) for child in root.children])
