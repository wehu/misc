import networkx as nx
import pymetis as pm
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# DSL
def variable(graph, name, type=None, shape=[]):
    #assert(type)
    #assert(shape != [])
    graph.add_node(
        name,
        type=type,
        shape=shape,
        weight=.0,
    )
    return name


def vertex(graph, inputs, outputs, cost, func):
    id = len(graph.nodes)
    graph.add_node(
        id,
        inputs=inputs,
        outputs=outputs,
        func=func,
        weight=cost,
    )
    return id


def connect(graph, vertex, port_name, var_name, slices):
    graph.nodes[vertex][port_name] = (var_name, slices)
    graph.add_edge(vertex, var_name)


# operator implementations
def matmul(graph, node):
    v = vertex(graph, [], [], 0.5, lambda x: x)
    for input in node.input:
        connect(graph, v, input, input, [])
    for output in node.output:
        connect(graph, v, output, output, [])


def add(graph, node):
    v = vertex(graph, [], [], 0.1, lambda x: x)
    for input in node.input:
        connect(graph, v, input, input, [])
    for output in node.output:
        connect(graph, v, output, output, [])


# convertor
def onnx2graph(model):
    g = nx.Graph()
    # inputs
    for node in model.graph.input:
        variable(g, node.name)
    # internal tensors
    for node in model.graph.value_info:
        variable(g, node.name)
    # outputs
    for node in model.graph.output:
        variable(g, node.name)
    # computing nodes
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            matmul(g, node)
        elif node.op_type == 'Add':
            add(g, node)
    return g


# merge nodes
def merge_nodes(graph):
    return graph


# placement and routing
def place_and_route(graph):
    return graph


# compile
def compile_onnx(model):
    g = onnx2graph(model)
    g = merge_nodes(g)
    g = place_and_route(g)
    return g


# test

# create onnx model
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [4, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 8])
T = helper.make_tensor_value_info('T', TensorProto.FLOAT, [4, 8])
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [4, 8])

node_matmul = helper.make_node(
    'MatMul',
    ['X', 'Y'],
    ['T'],
)

node_add = helper.make_node(
    'Add',
    ['T', 'T'],
    ['Z'],
)

graph_def = helper.make_graph(
    [node_matmul, node_add],
    'test-model',
    [X, Y],
    [Z],
)

model_def = helper.make_model(graph_def, producer_name='matmul-add')
onnx.checker.check_model(model_def)

graph = compile_onnx(model_def)

nx.draw(graph)
plt.show()
