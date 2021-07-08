from antlr4.FileStream import FileStream
from pycode2seq.inference.common.node import Node
from typing import Callable

from antlr4 import Lexer, InputStream, Parser as AntlrParser, TokenStream, ParserRuleContext

from pycode2seq.inference.parsing.parser import Parser
from pycode2seq.inference.antlr.Java8Lexer import Java8Lexer
from pycode2seq.inference.antlr.Java8Parser import Java8Parser
from pycode2seq.inference.antlr.KotlinLexer import KotlinLexer
from pycode2seq.inference.antlr.KotlinParser import KotlinParser as AntlrKotlinParser


class KotlinParser(Parser):
    def _lexer(self, input_stream: InputStream) -> Lexer:
        return KotlinLexer(input_stream)

    def _parser(self, stream: TokenStream) -> AntlrParser:
        return AntlrKotlinParser(stream)

    def _get_tree(self, parser: AntlrParser) -> ParserRuleContext:
        if not isinstance(parser, AntlrKotlinParser):
            raise ValueError("Wrong parser")
        return parser.kotlinFile()


class JavaParser(Parser):
    def _lexer(self, input_stream: InputStream) -> Lexer:
        return Java8Lexer(input_stream)

    def _parser(self, stream: TokenStream) -> AntlrParser:
        return Java8Parser(stream)

    def _get_tree(self, parser: AntlrParser) -> ParserRuleContext:
        if not isinstance(parser, Java8Parser):
            raise ValueError("Wrong parser")
        return parser.compilationUnit()


class SpeedyKotlinParser(KotlinParser):
    def parse(self, file_path: str) -> Node:
        from spam.parser import sa_kotlin
        input_stream = FileStream(file_path)
        lexer = self._lexer(input_stream)
        input_stream = FileStream(file_path)
        tree = sa_kotlin.parse(input_stream, "kotlinFile", sa_kotlin.SA_ErrorListener())
        return Parser._compress_tree(Parser._convert_rule_context(tree, self._parser(None).ruleNames, None, lexer))