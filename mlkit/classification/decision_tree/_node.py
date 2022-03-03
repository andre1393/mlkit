class Node:
    
    def __init__(self, parent, children, attribute_name):
        self.children = {}
        self.link = parent
        if children is not None:
            self.children[parent] = children
        self.attribute_name = attribute_name
        
    def add_node(self, children_name, node):
        self.children[children_name] = node
