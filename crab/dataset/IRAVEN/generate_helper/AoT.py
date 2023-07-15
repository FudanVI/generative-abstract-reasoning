import copy
from dataset.RAVEN.generate_helper.Attribute import \
    Angle, Color, Number, Position, Size, Type, Uniformity


class AoTNode(object):
    levels_next = {"Root": "Structure",
                   "Structure": "Component",
                   "Component": "Layout",
                   "Layout": "Entity"}

    def __init__(self, name, level, node_type):
        self.name = name
        self.level = level
        self.node_type = node_type
        self.children = []

    def insert(self, node):
        """Used for private.
        Arguments:
            node(AoTNode): a node to insert
        """
        assert isinstance(node, AoTNode)
        assert self.node_type != "leaf"
        assert node.level == self.levels_next[self.level]
        self.children.append(node)

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


class Root(AoTNode):

    def __init__(self, name):
        super(Root, self).__init__(name, level="Root", node_type="or")

    def sample(self):
        """The function returns a separate AoT that is correctly parsed.
        Note that a new node is needed so that modification does not alter settings
        in the original tree.
        Returns:
            new_node(Root): a newly instantiated node
        """
        self.children[0].sample()

    def prepare(self):
        """This function prepares the AoT for rendering.
        Returns:
            structure.name(str): used for rendering structure
            entities(list of Entity): used for rendering each entity
        """
        structure = self.children[0]
        components = []
        for child in structure.children:
            components.append(child)
        entities = []
        for component in components:
            for child in component.children[0].children:
                entities.append(child)
        return structure.name, entities

    def set_attr(self, component_idx, attr_name, levels):
        self.children[0].set_attr(component_idx, attr_name, levels)

    def get_attr_level(self, component_idx, attr_name):
        return self.children[0].get_attr_level(component_idx, attr_name)


class Structure(AoTNode):

    def __init__(self, name):
        super(Structure, self).__init__(name, level="Structure", node_type="and")

    def sample(self):
        for child in self.children:
            child.sample()

    def set_attr(self, component_idx, attr_name, levels):
        self.children[component_idx].set_attr(attr_name, levels)

    def get_attr_level(self, component_idx, attr_name):
        return self.children[component_idx].get_attr_level(attr_name)


class Component(AoTNode):

    def __init__(self, name):
        super(Component, self).__init__(name, level="Component", node_type="or")

    def sample(self):
        self.children[0].sample()

    def set_attr(self, attr_name, levels):
        self.children[0].set_attr(attr_name, levels)

    def get_attr_level(self, attr_name):
        return self.children[0].get_attr_level(attr_name)


class Layout(AoTNode):

    def __init__(self, name, layout_constraint, entity_constraint):
        super(Layout, self).__init__(name, level="Layout", node_type="and")
        self.layout_constraint = layout_constraint
        self.entity_constraint = entity_constraint
        self.number = Number(
            min_level=layout_constraint["Number"][0],
            max_level=layout_constraint["Number"][1]
        )
        self.position = Position(
            pos_type=layout_constraint["Position"][0],
            pos_list=layout_constraint["Position"][1]
        )
        self.uniformity = Uniformity(
            min_level=layout_constraint["Uni"][0],
            max_level=layout_constraint["Uni"][1]
        )
        self.number.sample()
        self.position.sample(self.number.get_value())
        self.uniformity.sample()

    def add_new(self, *bboxes):
        """Add new entities into this level.
        Arguments:
            *bboxes(tuple of bbox): bboxes of new entities
        """
        name = self.number.get_value()
        uni = self.uniformity.get_value()
        for i in range(len(bboxes)):
            name += i
            bbox = bboxes[i]
            new_entity = copy.deepcopy(self.children[0])
            new_entity.name = str(name)
            new_entity.bbox = bbox
            if not uni:
                new_entity.resample()
            self.insert(new_entity)

    def sample(self):
        self.number.sample()
        self.position.sample(self.number.get_value())
        pos = self.position.get_value()
        del self.children[:]
        if self.uniformity.get_value():
            node = Entity(name=str(0), bbox=pos[0], entity_constraint=self.entity_constraint)
            self.insert(node)
            for i in range(1, len(pos)):
                bbox = pos[i]
                node = copy.deepcopy(node)
                node.name = str(i)
                node.bbox = bbox
                self.insert(node)
        else:
            for i in range(len(pos)):
                bbox = pos[i]
                node = Entity(name=str(i), bbox=bbox, entity_constraint=self.entity_constraint)
                self.insert(node)

    def set_attr(self, attr_name, levels):
        if attr_name == "Num/Pos":
            self.number.set_value_level(levels[0])
            self.position.set_value_idx(levels[1])
            pos = self.position.get_value()
            del self.children[:]
            for i in range(len(pos)):
                bbox = pos[i]
                node = Entity(name=str(i), bbox=bbox, entity_constraint=self.entity_constraint)
                self.insert(node)
        elif attr_name == "Type":
            for index in range(len(self.children)):
                self.children[index].type.set_value_level(levels[0])
        elif attr_name == "Size":
            for index in range(len(self.children)):
                self.children[index].size.set_value_level(levels[0])
        elif attr_name == "Color":
            for index in range(len(self.children)):
                self.children[index].color.set_value_level(levels[0])
        else:
            raise ValueError("Unsupported operation")

    def get_attr_level(self, attr_name):
        if attr_name == "Num/Pos":
            return [self.number.get_value_level(), self.position.get_value_idx()]
        elif attr_name == "Type":
            return [self.children[0].type.get_value_level()]
        elif attr_name == "Size":
            return [self.children[0].size.get_value_level()]
        elif attr_name == "Color":
            return [self.children[0].color.get_value_level()]
        else:
            raise ValueError("Unsupported operation")


class Entity(AoTNode):

    def __init__(self, name, bbox, entity_constraint):
        super(Entity, self).__init__(name, level="Entity", node_type="leaf")
        # Attributes
        # Sample each attribute such that the value lies in the admissible range
        # Otherwise, random sample
        self.entity_constraint = entity_constraint
        self.bbox = bbox
        self.type = Type(
            min_level=entity_constraint["Type"][0],
            max_level=entity_constraint["Type"][1]
        )
        self.type.sample()
        self.size = Size(
            min_level=entity_constraint["Size"][0],
            max_level=entity_constraint["Size"][1]
        )
        self.size.sample()
        self.color = Color(
            min_level=entity_constraint["Color"][0],
            max_level=entity_constraint["Color"][1]
        )
        self.color.sample()
        self.angle = Angle(
            min_level=entity_constraint["Angle"][0],
            max_level=entity_constraint["Angle"][1]
        )
        self.angle.sample()

    def re_sample(self):
        self.type.sample()
        self.size.sample()
        self.color.sample()
        self.angle.sample()
