class Node:
    
    def __init__(self, link, children, attribute_name):
        self.children = {}
        self.link = link
        if children != None:
            self.children[link] = children
        self.attribute_name = attribute_name
        
    def add_node(self, children_name, node):
        self.children[children_name] = node