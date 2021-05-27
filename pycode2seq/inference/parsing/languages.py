from pycode2seq.inference.parsing.utils import compress_tree, convert_rule_context, shorten_nodes
from pycode2seq.inference.common.node import Node
from pycode2seq.inference.antlr.Java8Lexer import Java8Lexer
from pycode2seq.inference.antlr.Java8Parser import Java8Parser
from pycode2seq.inference.antlr.KotlinLexer import KotlinLexer
from pycode2seq.inference.antlr.KotlinParser import KotlinParser
from antlr4 import *

def parse_kotlin_file(file_path: str) -> Node:
    input_stream = FileStream(file_path)
    lexer = KotlinLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = KotlinParser(stream)
    tree = parser.kotlinFile()

    return compress_tree(convert_rule_context(tree, parser.ruleNames, None, lexer))

def parse_java_file(file_path: str) -> Node:
    input_stream = FileStream(file_path)
    lexer = Java8Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Java8Parser(stream)
    
    tree = parser.compilationUnit()
    return compress_tree(convert_rule_context(tree, parser.ruleNames, None, lexer))
